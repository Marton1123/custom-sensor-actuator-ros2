#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 para control de actuadores, ciclo cerrado de reciclaje (Limpiaparabrisas) y
validacion de peso mediante celda de carga (HX711) sobre Raspberry Pi 5 (lgpio).

Arquitectura DevSecOps:
    - Controladores de hardware desacoplados con timeout estricto.
    - Ejecucion de la maquina de estados en worker thread para evitar bloqueo del executor ROS 2.
    - Sensor Fusion: Habilitacion de reciclaje condicionada a validacion de peso HX711 (Peso <= Umbral).
    - Ciclo Cerrado de Reciclaje (Expulsion -> Retorno Logico -> Anclaje IR continuo) sin comandos intermedios.
"""

import time
import threading
from typing import Optional, Tuple

try:
    import lgpio
    _LGPIO_DISPONIBLE = True
except ImportError:
    lgpio = None  # type: ignore[assignment]
    _LGPIO_DISPONIBLE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32, String


# ============================================================================
# PARAMETROS Y CONSTANTES GLOBALES DE CALIBRACION
# ============================================================================
# Calibracion empirica de celda de carga HX711
DEFAULT_HX711_OFFSET: float = 0.0
DEFAULT_HX711_REFERENCE_UNIT: float = 2273.9  # Factor ADC -> gramos
DEFAULT_UMBRAL_PESO_MAX: float = 40.0         # Gramos (envase limpio)

# Parametros de dinamica de actuadores y ciclo de reciclaje
DEFAULT_TIEMPO_EXTRA_EXPULSION: float = 0.8   # Segundos de giro tras detectar 1-1 en expulsion
DEFAULT_PULSO_INERCIA: float = 0.005          # Pulso de gracia / inercia (5 ms)
DEFAULT_TIMEOUT_CICLO: float = 7.0            # Timeout maximo de ciclo completo (7 s)

# Definiciones de estados de sentido de giro
DIR_DERECHA: Tuple[int, int, int, int] = (1, 0, 1, 0)
DIR_IZQUIERDA: Tuple[int, int, int, int] = (0, 1, 0, 1)
DIR_STOP: Tuple[int, int, int, int] = (0, 0, 0, 0)


class HX711DriverLGPIO:
    """
    Controlador de bajo nivel para convertidor ADC HX711 mediante bit-banging en lgpio.
    Diseñado para operar en Raspberry Pi 5 sin bloqueos indefinidos de hardware.
    """

    def __init__(self, handle: int, dt_pin: int, sck_pin: int, gain: int = 128) -> None:
        self._h = handle
        self._dt = dt_pin
        self._sck = sck_pin
        self._gain = gain
        self._offset: float = DEFAULT_HX711_OFFSET

        if self._gain == 128:
            self._extra_pulses = 1
        elif self._gain == 64:
            self._extra_pulses = 3
        elif self._gain == 32:
            self._extra_pulses = 2
        else:
            self._extra_pulses = 1

    def is_ready(self, timeout: float = 0.3) -> bool:
        """
        Espera no bloqueante a que el pin DT baje a LOW.
        """
        t0 = time.time()
        while lgpio.gpio_read(self._h, self._dt) != 0:
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.001)
        return True

    def read_raw(self) -> int:
        """
        Lee 24 bits crudos en complemento a dos desde el HX711.
        """
        if not self.is_ready():
            raise TimeoutError("Timeout en comunicacion con hardware HX711")

        value = 0
        for _ in range(24):
            lgpio.gpio_write(self._h, self._sck, 1)
            lgpio.gpio_write(self._h, self._sck, 0)
            bit = lgpio.gpio_read(self._h, self._dt)
            value = (value << 1) | (bit & 1)

        for _ in range(self._extra_pulses):
            lgpio.gpio_write(self._h, self._sck, 1)
            lgpio.gpio_write(self._h, self._sck, 0)

        if value & 0x800000:
            value -= 0x1000000

        return value

    def read_average(self, times: int = 3) -> float:
        """
        Toma multiples muestras descartando errores de timeout.
        """
        suma = 0.0
        exitos = 0
        for _ in range(times):
            try:
                suma += self.read_raw()
                exitos += 1
            except TimeoutError:
                pass
            time.sleep(0.005)

        if exitos == 0:
            raise RuntimeError("Fallo de lectura en celda de carga HX711")

        return suma / exitos

    def set_offset(self, offset: float) -> None:
        self._offset = offset

    def get_offset(self) -> float:
        return self._offset

    def get_weight(self, reference_unit: float, times: int = 3) -> float:
        """
        Calcula el peso neto en gramos aplicando tara y factor de escala.
        """
        raw = self.read_average(times=times)
        if reference_unit == 0.0:
            raise ValueError("reference_unit no puede ser cero")
        return (raw - self._offset) / reference_unit


class NodoActuadores(Node):
    """
    Nodo ROS 2 orquestador de actuadores y Sensor Fusion (IR + HX711).
    """

    TOPIC_COMANDO_GRADOS = "/comando_grados"
    TOPIC_OBJETO_DETECTADO = "/objeto_detectado"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        if not _LGPIO_DISPONIBLE:
            self.get_logger().fatal(
                "lgpio no esta disponible. El nodo_actuadores no puede arrancar."
            )
            raise RuntimeError("lgpio no disponible")

        # ── Declaracion de Parametros ROS 2 ─────────────────────────────────
        self.declare_parameter("gpio_chip", 4)
        self.declare_parameter("tiempo_extra_expulsion", DEFAULT_TIEMPO_EXTRA_EXPULSION)
        self.declare_parameter("umbral_peso_max", DEFAULT_UMBRAL_PESO_MAX)
        self.declare_parameter("hx711_offset", DEFAULT_HX711_OFFSET)
        self.declare_parameter("hx711_reference_unit", DEFAULT_HX711_REFERENCE_UNIT)
        self.declare_parameter("pulso_inercia", DEFAULT_PULSO_INERCIA)
        self.declare_parameter("timeout_ciclo", DEFAULT_TIMEOUT_CICLO)

        self._gpio_chip = self.get_parameter("gpio_chip").value
        self._tiempo_extra_expulsion = float(self.get_parameter("tiempo_extra_expulsion").value)
        self._umbral_peso_max = float(self.get_parameter("umbral_peso_max").value)
        self._hx711_offset = float(self.get_parameter("hx711_offset").value)
        self._hx711_reference_unit = float(self.get_parameter("hx711_reference_unit").value)
        self._pulso_inercia = float(self.get_parameter("pulso_inercia").value)
        self._timeout_ciclo = float(self.get_parameter("timeout_ciclo").value)

        # ── Asignacion de Pines GPIO (BCM) ──────────────────────────────────
        # Driver Motores L298N
        self.IN1 = 18
        self.IN2 = 23
        self.IN3 = 25
        self.IN4 = 26

        # Sensores Infrarrojos QTR-1A (Micro-Centrado)
        self.SENSOR_IZQ = 17
        self.SENSOR_DER = 27

        # Celda de Carga (HX711)
        self.HX711_DT = 5
        self.HX711_SCK = 6

        # ── Concurrencia y Sincronizacion ──────────────────────────────────
        self._hilo_ciclo: Optional[threading.Thread] = None
        self._cancelar = threading.Event()
        self._lock = threading.Lock()

        # ── Inicializacion de Hardware GPIO ────────────────────────────────
        self._h = self._init_gpio()
        self._hx711 = HX711DriverLGPIO(self._h, self.HX711_DT, self.HX711_SCK)
        self._hx711.set_offset(self._hx711_offset)

        # ── QoS y Suscripciones ───────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._sub_grados = self.create_subscription(
            Float32,
            self.TOPIC_COMANDO_GRADOS,
            self._cb_comando_grados,
            qos,
        )

        self._sub_vision = self.create_subscription(
            String,
            self.TOPIC_OBJETO_DETECTADO,
            self._cb_objeto_detectado,
            qos,
        )

        self.get_logger().info(
            f"NodoActuadores inicializado correctamente\n"
            f"  GPIO Chip        : gpiochip{self._gpio_chip}\n"
            f"  Motores          : M1(IN1={self.IN1}, IN2={self.IN2}), M2(IN3={self.IN3}, IN4={self.IN4})\n"
            f"  Sensores IR      : Izq=GPIO{self.SENSOR_IZQ}, Der=GPIO{self.SENSOR_DER} (PULL_UP)\n"
            f"  HX711 Balanza    : DT=GPIO{self.HX711_DT}, SCK=GPIO{self.HX711_SCK} (Umbral: {self._umbral_peso_max}g)\n"
            f"  Timeout Ciclo    : {self._timeout_ciclo}s\n"
            f"  Suscripciones    : '{self.TOPIC_COMANDO_GRADOS}', '{self.TOPIC_OBJETO_DETECTADO}'"
        )

    def _init_gpio(self) -> int:
        """
        Reclama y configura todos los pines GPIO de forma segura.
        Los sensores IR se configuran obligatoriamente con resistencia interna PULL_UP.
        """
        handle = lgpio.gpiochip_open(self._gpio_chip)

        # Salidas de motores (inicializadas en LOW)
        lgpio.gpio_claim_output(handle, self.IN1, 0)
        lgpio.gpio_claim_output(handle, self.IN2, 0)
        lgpio.gpio_claim_output(handle, self.IN3, 0)
        lgpio.gpio_claim_output(handle, self.IN4, 0)

        # Entradas digitales de sensores IR con PULL_UP
        lgpio.gpio_claim_input(handle, self.SENSOR_IZQ, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(handle, self.SENSOR_DER, lgpio.SET_PULL_UP)

        # Pines de comunicacion HX711
        lgpio.gpio_claim_input(handle, self.HX711_DT)
        lgpio.gpio_claim_output(handle, self.HX711_SCK, 0)

        return handle

    def _cb_comando_grados(self, msg: Float32) -> None:
        """
        Procesa comandos por angulo.
        El ciclo cerrado maneja el retorno a 0 de forma autonoma;
        los comandos intermedios de 0.0 recibidos durante la ejecucion se ignoran.
        """
        comando = msg.data
        self.get_logger().info(f"Comando grados recibido: {comando}")

        if comando == 90.0:
            self._lanzar_ciclo_reciclaje("GIRAR_BOTELLA")
        elif comando == -90.0:
            self._lanzar_ciclo_reciclaje("GIRAR_LATA")
        elif comando == 0.0:
            with self._lock:
                en_ejecucion = self._hilo_ciclo is not None and self._hilo_ciclo.is_alive()
            if en_ejecucion:
                self.get_logger().info(
                    "Comando 0.0 ignorado: el ciclo cerrado realiza el retorno y centrado automaticamente."
                )
            else:
                self.get_logger().info("Comando 0.0 recibido en estado de reposo.")
        else:
            self.get_logger().warning(f"Comando angular desconocido: {comando}")

    def _cb_objeto_detectado(self, msg: String) -> None:
        """
        Procesa detecciones directas de vision artificial.
        """
        objeto = msg.data.lower()
        if not objeto:
            return

        if objeto == "bottle":
            self.get_logger().info("Deteccion vision: BOTELLA -> Iniciando ciclo.")
            self._lanzar_ciclo_reciclaje("GIRAR_BOTELLA")
        elif objeto == "can":
            self.get_logger().info("Deteccion vision: LATA -> Iniciando ciclo.")
            self._lanzar_ciclo_reciclaje("GIRAR_LATA")

    def _lanzar_ciclo_reciclaje(self, accion: str) -> None:
        """
        Cancela cualquier movimiento previo y ejecuta el ciclo de lazo cerrado en un worker thread.
        """
        self._cancelar.set()

        with self._lock:
            hilo_anterior = self._hilo_ciclo

        if hilo_anterior and hilo_anterior.is_alive():
            hilo_anterior.join(timeout=2.0)

        with self._lock:
            self._cancelar.clear()
            self._hilo_ciclo = threading.Thread(
                target=self._maquina_estados_reciclaje,
                args=(accion,),
                daemon=True,
                name=f"ciclo_{accion}",
            )
            self._hilo_ciclo.start()

    def _maquina_estados_reciclaje(self, accion: str) -> None:
        """
        Maquina de estados para Sensor Fusion y ejecucion del ciclo cerrado de reciclaje.
        """
        self.get_logger().info(f"Iniciando ciclo cerrado de reciclaje para {accion}")

        # 1. Validacion de Limpieza con Balanza HX711
        self.get_logger().info("Validando peso del envase con celda de carga...")
        limpio_ok, peso_medido = self._validar_peso()
        if not limpio_ok:
            self.get_logger().warning(
                f"Rechazado: Envase contaminado (Peso={peso_medido:.2f}g > Umbral={self._umbral_peso_max}g)."
            )
            self._frenar_motores()
            return

        self.get_logger().info(
            f"Sensor Fusion OK (Peso={peso_medido:.2f}g <= {self._umbral_peso_max}g). Habilitando ciclo de expulsion."
        )

        # 2. Mapeo de direccion de expulsion inicial
        if accion == "GIRAR_LATA":
            direccion_ida = DIR_IZQUIERDA
        elif accion == "GIRAR_BOTELLA":
            direccion_ida = DIR_DERECHA
        else:
            self.get_logger().error(f"Accion desconocida: {accion}")
            return

        # 3. Ejecucion del ciclo ininterrumpido de Ida y Vuelta
        exito = self._ejecutar_ciclo_expulsion(direccion_ida)
        if exito:
            self.get_logger().info(f"Ciclo cerrado de reciclaje ({accion}) completado exitosamente.")
        else:
            self.get_logger().warning(f"Ciclo cerrado de reciclaje ({accion}) finalizo con errores o timeout.")

    def _ejecutar_ciclo_expulsion(
        self, direccion_ida: Tuple[int, int, int, int]
    ) -> bool:
        """
        Ejecuta el ciclo cerrado de reciclaje ininterrumpido (Limpiaparabrisas):
            1. Fase de Expulsion: Giro continuo en direccion_ida hasta detectar 1-1 + delay de 0.8s.
            2. Fase de Retorno Logico: Inversion de giro continua hacia el centro hasta detectar borde blanco.
            3. Fase de Anclaje: Micro-centrado continuo hasta 0-0 con pulso de inercia y frenado.
        """
        t_inicio = time.time()
        timeout = self._timeout_ciclo
        pulso_inercia = self._pulso_inercia
        tiempo_extra = self._tiempo_extra_expulsion

        # Definicion de direccion inversa para el retorno
        dir_retorno = DIR_IZQUIERDA if direccion_ida == DIR_DERECHA else DIR_DERECHA

        # ────────────────────────────────────────────────────────────────────
        # FASE 1: EXPULSION (Escape del blanco hacia el contenedor)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 1: Expulsion activa hacia contenedor...")
        self._aplicar_estado_motores(*direccion_ida)

        # Esperar a que el brazo salga de la cinta blanca (ambos sensores en 1)
        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            if izq == 1 and der == 1:
                # El brazo ha salido del centro; mantener giro el tiempo extra de seguridad
                self.get_logger().info("Brazo fuera de rango central (1-1). Aplicando delay de caida...")
                t_delay = time.time()
                while (time.time() - t_delay) < tiempo_extra:
                    if self._cancelar.is_set():
                        self._frenar_motores()
                        return False
                    time.sleep(0.01)
                break

            time.sleep(0.001)
        else:
            self.get_logger().warning("Timeout en Fase 1 (Expulsion). Abortando ciclo.")
            self._frenar_motores()
            return False

        # ────────────────────────────────────────────────────────────────────
        # FASE 2: RETORNO LOGICO (Busqueda del cero con orientacion conocida)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 2: Retorno continuo buscando borde central...")
        self._aplicar_estado_motores(*dir_retorno)

        # Retornar continuamente hasta que cualquier sensor toque la cinta blanca (deje de ser 1-1)
        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            if izq == 0 or der == 0:
                self.get_logger().info(f"Borde central detectado (Izq={izq}, Der={der}). Pasando a Fase 3.")
                break

            time.sleep(0.001)
        else:
            self.get_logger().warning("Timeout en Fase 2 (Retorno). Abortando ciclo.")
            self._frenar_motores()
            return False

        # ────────────────────────────────────────────────────────────────────
        # FASE 3: ANCLAJE (Micro-centrado continuo de precision)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 3: Micro-centrado y anclaje continuo...")
        ultima_direccion = dir_retorno

        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            # Estado 0-0: Centro perfecto alcanzado
            if izq == 0 and der == 0:
                self.get_logger().info(
                    "Centro perfecto alcanzado (0-0). Aplicando compensacion de inercia y frenado."
                )
                self._aplicar_estado_motores(*ultima_direccion)
                time.sleep(pulso_inercia)
                self._frenar_motores()
                return True

            if izq == 0 and der == 1:
                # Desviado a la izquierda -> Corregir a la derecha
                ultima_direccion = DIR_DERECHA
                self._aplicar_estado_motores(*DIR_DERECHA)
            elif izq == 1 and der == 0:
                # Desviado a la derecha -> Corregir a la izquierda
                ultima_direccion = DIR_IZQUIERDA
                self._aplicar_estado_motores(*DIR_IZQUIERDA)
            else:
                # 1-1 durante correccion -> Mantener ultima direccion conocida
                self._aplicar_estado_motores(*ultima_direccion)

            time.sleep(0.001)

        self.get_logger().warning("Timeout en Fase 3 (Anclaje). Abortando ciclo.")
        self._frenar_motores()
        return False

    def _validar_peso(self) -> Tuple[bool, float]:
        """
        Lee el peso del envase y valida contra el umbral permitido.
        """
        try:
            peso = self._hx711.get_weight(self._hx711_reference_unit, times=3)
            self.get_logger().info(f"Lectura de peso: {peso:.2f} g")
            if peso <= self._umbral_peso_max:
                return True, peso
            return False, peso
        except Exception as exc:
            self.get_logger().error(f"Error leyendo sensor HX711: {exc}")
            return False, 999.0

    def _aplicar_estado_motores(self, m1_in1: int, m1_in2: int, m2_in3: int, m2_in4: int) -> None:
        """
        Aplica directamente los niveles logicos a los pines de los motores sin esperas.
        """
        lgpio.gpio_write(self._h, self.IN1, m1_in1)
        lgpio.gpio_write(self._h, self.IN2, m1_in2)
        lgpio.gpio_write(self._h, self.IN3, m2_in3)
        lgpio.gpio_write(self._h, self.IN4, m2_in4)

    def _frenar_motores(self) -> None:
        """
        Detiene instantaneamente ambos motores poniendo todos los pines en LOW.
        """
        lgpio.gpio_write(self._h, self.IN1, 0)
        lgpio.gpio_write(self._h, self.IN2, 0)
        lgpio.gpio_write(self._h, self.IN3, 0)
        lgpio.gpio_write(self._h, self.IN4, 0)

    def destroy_node(self) -> None:
        """
        Detiene actuadores y libera todos los recursos GPIO de forma segura.
        """
        self.get_logger().info("Apagando NodoActuadores y liberando pines GPIO...")
        self._cancelar.set()

        with self._lock:
            hilo = self._hilo_ciclo
        if hilo and hilo.is_alive():
            hilo.join(timeout=2.0)

        try:
            self._frenar_motores()

            # Liberacion de pines de actuadores
            lgpio.gpio_free(self._h, self.IN1)
            lgpio.gpio_free(self._h, self.IN2)
            lgpio.gpio_free(self._h, self.IN3)
            lgpio.gpio_free(self._h, self.IN4)

            # Liberacion de sensores IR
            lgpio.gpio_free(self._h, self.SENSOR_IZQ)
            lgpio.gpio_free(self._h, self.SENSOR_DER)

            # Liberacion de sensor HX711
            lgpio.gpio_free(self._h, self.HX711_DT)
            lgpio.gpio_free(self._h, self.HX711_SCK)

            lgpio.gpiochip_close(self._h)
            self.get_logger().info("Recursos GPIO cerrados correctamente.")
        except Exception as exc:
            self.get_logger().error(f"Error al cerrar GPIO: {exc}")

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        nodo = NodoActuadores()
    except RuntimeError:
        rclpy.shutdown()
        return
    except Exception as exc:
        import logging
        logging.getLogger(__name__).fatal(
            f"Error fatal inicializando NodoActuadores: {exc}"
        )
        rclpy.shutdown()
        return

    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
