#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 para control de actuadores, micro-centrado IR (Bang-Bang) y
validacion de peso mediante celda de carga (HX711) sobre Raspberry Pi 5 (lgpio).

Arquitectura DevSecOps:
    - Controladores de hardware desacoplados con timeout estricto.
    - Ejecucion de la maquina de estados en worker thread para evitar bloqueo del executor ROS 2.
    - Sensor Fusion: Habilitacion de reciclaje condicionada a (Centrado IR == OK) AND (Peso <= Umbral).
    - Centrado IR QTR-1A con Pull-Up interno, logica invertida y compensacion de inercia (backlash).
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

# Parametros de tiempo y dinamica de actuadores
DEFAULT_TIEMPO_MOTOR: float = 4.7
DEFAULT_PULSO_MICROCENTRADO: float = 0.05      # Rafaga normal (50 ms)
DEFAULT_PULSO_INERCIA: float = 0.005           # Pulso de gracia / inercia (5 ms)
DEFAULT_TIMEOUT_MICROCENTRADO: float = 1.5


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
        self.declare_parameter("tiempo_motor", DEFAULT_TIEMPO_MOTOR)
        self.declare_parameter("umbral_peso_max", DEFAULT_UMBRAL_PESO_MAX)
        self.declare_parameter("hx711_offset", DEFAULT_HX711_OFFSET)
        self.declare_parameter("hx711_reference_unit", DEFAULT_HX711_REFERENCE_UNIT)
        self.declare_parameter("pulso_microcentrado", DEFAULT_PULSO_MICROCENTRADO)
        self.declare_parameter("pulso_inercia", DEFAULT_PULSO_INERCIA)
        self.declare_parameter("timeout_microcentrado", DEFAULT_TIMEOUT_MICROCENTRADO)

        self._gpio_chip = self.get_parameter("gpio_chip").value
        self._tiempo_motor = float(self.get_parameter("tiempo_motor").value)
        self._umbral_peso_max = float(self.get_parameter("umbral_peso_max").value)
        self._hx711_offset = float(self.get_parameter("hx711_offset").value)
        self._hx711_reference_unit = float(self.get_parameter("hx711_reference_unit").value)
        self._pulso_microcentrado = float(self.get_parameter("pulso_microcentrado").value)
        self._pulso_inercia = float(self.get_parameter("pulso_inercia").value)
        self._timeout_microcentrado = float(self.get_parameter("timeout_microcentrado").value)

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
            f"  GPIO Chip    : gpiochip{self._gpio_chip}\n"
            f"  Motores      : M1(IN1={self.IN1}, IN2={self.IN2}), M2(IN3={self.IN3}, IN4={self.IN4})\n"
            f"  Sensores IR  : Izq=GPIO{self.SENSOR_IZQ}, Der=GPIO{self.SENSOR_DER} (PULL_UP)\n"
            f"  HX711 Balanza: DT=GPIO{self.HX711_DT}, SCK=GPIO{self.HX711_SCK} (Umbral: {self._umbral_peso_max}g)\n"
            f"  Suscripciones: '{self.TOPIC_COMANDO_GRADOS}', '{self.TOPIC_OBJETO_DETECTADO}'"
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
        """
        comando = msg.data
        self.get_logger().info(f"Comando grados recibido: {comando}")

        if comando == 90.0:
            self._lanzar_ciclo_reciclaje("GIRAR_BOTELLA")
        elif comando == -90.0:
            self._lanzar_ciclo_reciclaje("GIRAR_LATA")
        elif comando == 0.0:
            self.get_logger().info("Comando Reset/Centro recibido.")
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
        Cancela cualquier movimiento previo y ejecuta la rutina completa en un worker thread.
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
        Maquina de estados para Sensor Fusion y ejecucion de reciclaje.
        """
        self.get_logger().info(f"Iniciando ciclo de reciclaje para {accion}")

        # 1. Movimiento de centrado grueso
        self.get_logger().info("Paso 1: Ejecutando centrado grueso...")
        if not self._ejecutar_centrado_grueso():
            self.get_logger().warning("Ciclo cancelado durante centrado grueso.")
            return

        # 2. Micro-centrado IR
        self.get_logger().info("Paso 2: Ejecutando micro-centrado IR...")
        centrado_ok = self._micro_centrar()
        if not centrado_ok:
            self.get_logger().warning("Rechazado: Falla en micro-centrado IR. Abortando reciclaje.")
            self._frenar_motores()
            return

        # 3. Validacion de Limpieza con Balanza HX711
        self.get_logger().info("Paso 3: Validando peso del envase con celda de carga...")
        limpio_ok, peso_medido = self._validar_peso()
        if not limpio_ok:
            self.get_logger().warning(
                f"Rechazado: Envase contaminado (Peso={peso_medido:.2f}g > Umbral={self._umbral_peso_max}g)."
            )
            self._frenar_motores()
            return

        # 4. Accion de Reciclaje Habilitada
        self.get_logger().info(
            f"Sensor Fusion OK (Centrado=OK, Peso={peso_medido:.2f}g). Habilitando reciclaje {accion}."
        )
        self._ejecutar_giro_reciclaje(accion)
        self.get_logger().info("Ciclo de reciclaje finalizado exitosamente.")

    def _ejecutar_centrado_grueso(self) -> bool:
        """
        Mueve los motores durante el tiempo configurado para la etapa gruesa.
        """
        tiempo = self._tiempo_motor
        lgpio.gpio_write(self._h, self.IN1, 1)
        lgpio.gpio_write(self._h, self.IN2, 0)
        lgpio.gpio_write(self._h, self.IN3, 1)
        lgpio.gpio_write(self._h, self.IN4, 0)

        t0 = time.time()
        while (time.time() - t0) < tiempo:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False
            time.sleep(0.01)

        self._frenar_motores()
        return True

    def _micro_centrar(self) -> bool:
        """
        Ejecuta control Bang-Bang con sensores IR (QTR-1A con Pull-Up) para centrado fino.
        Tabla de verdad (Blanco=0, Negro=1):
            0-0: Centro perfecto -> Aplica pulso de inercia y frena.
            0-1: Desviado a la izquierda -> Corrige a la derecha.
            1-0: Desviado a la derecha -> Corrige a la izquierda.
            1-1: Brazo no detectado -> Busqueda a ciegas hacia la derecha.
        """
        inicio = time.time()
        timeout = self._timeout_microcentrado
        pulso_ms = self._pulso_microcentrado
        pulso_inercia = self._pulso_inercia

        dir_derecha = (1, 0, 1, 0)
        dir_izquierda = (0, 1, 0, 1)
        ultima_direccion = dir_derecha

        while (time.time() - inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            # Estado 0-0: Centro perfecto
            if izq == 0 and der == 0:
                self.get_logger().info(
                    "Micro-centrado exitoso (0-0): Aplicando compensacion de inercia."
                )
                self._aplicar_pulso_motor(*ultima_direccion, pulso_inercia)
                self._frenar_motores()
                return True

            if izq == 0 and der == 1:
                # Brazo desviado a la izquierda -> Corregir a la derecha
                ultima_direccion = dir_derecha
                self._aplicar_pulso_motor(*dir_derecha, pulso_ms)
                time.sleep(0.1)
            elif izq == 1 and der == 0:
                # Brazo desviado a la derecha -> Corregir a la izquierda
                ultima_direccion = dir_izquierda
                self._aplicar_pulso_motor(*dir_izquierda, pulso_ms)
                time.sleep(0.1)
            else:  # 1-1: Brazo perdido
                # Busqueda a ciegas hacia la derecha
                ultima_direccion = dir_derecha
                self._aplicar_pulso_motor(*dir_derecha, pulso_ms)
                time.sleep(0.1)

        self.get_logger().warning("Micro-centrado fallido: Timeout superado.")
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

    def _ejecutar_giro_reciclaje(self, accion: str) -> None:
        """
        Ejecuta la apertura/giro clasificador definitivo.
        """
        tiempo = self._tiempo_motor
        if accion == "GIRAR_BOTELLA":
            lgpio.gpio_write(self._h, self.IN1, 1)
            lgpio.gpio_write(self._h, self.IN2, 0)
            lgpio.gpio_write(self._h, self.IN3, 1)
            lgpio.gpio_write(self._h, self.IN4, 0)
        elif accion == "GIRAR_LATA":
            lgpio.gpio_write(self._h, self.IN1, 0)
            lgpio.gpio_write(self._h, self.IN2, 1)
            lgpio.gpio_write(self._h, self.IN3, 0)
            lgpio.gpio_write(self._h, self.IN4, 1)

        t0 = time.time()
        while (time.time() - t0) < tiempo:
            if self._cancelar.is_set():
                break
            time.sleep(0.01)

        self._frenar_motores()

    def _aplicar_pulso_motor(
        self, m1_in1: int, m1_in2: int, m2_in3: int, m2_in4: int, duracion: float
    ) -> None:
        lgpio.gpio_write(self._h, self.IN1, m1_in1)
        lgpio.gpio_write(self._h, self.IN2, m1_in2)
        lgpio.gpio_write(self._h, self.IN3, m2_in3)
        lgpio.gpio_write(self._h, self.IN4, m2_in4)
        time.sleep(duracion)
        self._frenar_motores()

    def _frenar_motores(self) -> None:
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
