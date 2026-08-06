#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 para control de actuadores, validacion de peso de envases (HX711)
y ciclo continuo unidireccional de 360 grados con micro-centrado IR (Raspberry Pi 5 / lgpio).

Arquitectura DevSecOps:
    - Controladores de hardware desacoplados con proteccion contra bloqueos.
    - Ejecucion de la maquina de estados en worker thread para preservar la reactividad del executor ROS 2.
    - Fase 0 (Compuerta Logica): Validacion de peso mediante celda de carga HX711 calibrada (Lata vs Botella).
    - Ciclo 360 Unidireccional: Escape -> Trayecto continuo -> Auto-ajuste de anclaje IR (0-0).
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
DEFAULT_HX711_SCALE: float = 2039.0          # Factor ADC -> gramos

# Umbrales maximos de peso de seguridad (Gramos)
DEFAULT_PESO_MAX_LATA: float = 40.0
DEFAULT_PESO_MAX_BOTELLA: float = 80.0

# Parametros de dinamica de actuadores y ciclo de reciclaje
DEFAULT_PULSO_INERCIA: float = 0.005         # Pulso de gracia / inercia (5 ms)
DEFAULT_TIMEOUT_CICLO: float = 20.0          # Timeout maximo de ciclo completo (20 s)

# Definiciones de estados de sentido de giro
DIR_DERECHA: Tuple[int, int, int, int] = (1, 0, 1, 0)
DIR_IZQUIERDA: Tuple[int, int, int, int] = (0, 1, 0, 1)
DIR_STOP: Tuple[int, int, int, int] = (0, 0, 0, 0)


class HX711:
    """
    Controlador optimizado para convertidor ADC HX711 sobre Raspberry Pi 5 mediante lgpio.
    Implementa bit-banging sincrono, tara y conversion a unidades con timeout no bloqueante.
    """

    def __init__(self, dt_pin: int, sck_pin: int, chip_handle: int) -> None:
        self.dt = dt_pin
        self.sck = sck_pin
        self.h = chip_handle
        lgpio.gpio_claim_output(self.h, self.sck)
        lgpio.gpio_write(self.h, self.sck, 0)
        lgpio.gpio_claim_input(self.h, self.dt)
        self.offset: float = 0.0
        self.scale: float = 1.0

    def is_ready(self, timeout: float = 0.5) -> bool:
        """
        Verifica si el pin DT bajo a nivel logico 0 (listo para lectura).
        """
        t0 = time.time()
        while lgpio.gpio_read(self.h, self.dt) != 0:
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.001)
        return True

    def read_raw(self) -> int:
        """
        Lee 24 bits de datos en complemento a dos desde el HX711.
        """
        if not self.is_ready():
            raise TimeoutError("Timeout en comunicacion con hardware HX711 (pin DT)")

        data = 0
        write = lgpio.gpio_write
        read = lgpio.gpio_read
        h, sck, dt = self.h, self.sck, self.dt

        for _ in range(24):
            write(h, sck, 1)
            data = (data << 1) | read(h, dt)
            write(h, sck, 0)

        # Pulso 25 para fijar ganancia a 128 canal A
        write(h, sck, 1)
        write(h, sck, 0)

        if data & 0x800000:
            data -= 0x1000000

        return data

    def tare(self, times: int = 15) -> None:
        """
        Establece el cero relativo del sensor promediando lecturas crudas.
        """
        suma = sum(self.read_raw() for _ in range(times))
        self.offset = suma / times

    def set_scale(self, scale: float) -> None:
        """
        Asigna el factor de escala de calibracion ADC -> Gramos.
        """
        self.scale = scale

    def get_units(self, times: int = 5) -> float:
        """
        Obtiene el peso neto en gramos promediado.
        """
        if self.scale == 0.0:
            raise ValueError("scale no puede ser cero")
        suma = sum(self.read_raw() - self.offset for _ in range(times))
        return (suma / times) / self.scale


class NodoActuadores(Node):
    """
    Nodo ROS 2 orquestador de actuadores y Sensor Fusion (Balanza HX711 + Sensores IR).
    """

    TOPIC_COMANDO_GRADOS = "/comando_grados"
    TOPIC_OBJETO_DETECTADO = "/objeto_detectado"
    TOPIC_PESO_ELEMENTO = "/peso_elemento"
    TOPIC_ESTADO_RECICLAJE = "/clasificacion_objeto"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        if not _LGPIO_DISPONIBLE:
            self.get_logger().fatal(
                "lgpio no esta disponible. El nodo_actuadores no puede arrancar."
            )
            raise RuntimeError("lgpio no disponible")

        # ── Declaracion de Parametros ROS 2 ─────────────────────────────────
        self.declare_parameter("gpio_chip", 4)
        self.declare_parameter("hx711_scale", DEFAULT_HX711_SCALE)
        self.declare_parameter("peso_max_lata", DEFAULT_PESO_MAX_LATA)
        self.declare_parameter("peso_max_botella", DEFAULT_PESO_MAX_BOTELLA)
        self.declare_parameter("pulso_inercia", DEFAULT_PULSO_INERCIA)
        self.declare_parameter("timeout_ciclo", DEFAULT_TIMEOUT_CICLO)

        self._gpio_chip = self.get_parameter("gpio_chip").value
        self._hx711_scale = float(self.get_parameter("hx711_scale").value)
        self._peso_max_lata = float(self.get_parameter("peso_max_lata").value)
        self._peso_max_botella = float(self.get_parameter("peso_max_botella").value)
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

        # ── Inicializacion de Hardware GPIO y Balanza HX711 ────────────────
        self._h = self._init_gpio()
        self.balanza = HX711(dt_pin=self.HX711_DT, sck_pin=self.HX711_SCK, chip_handle=self._h)
        self.balanza.set_scale(self._hx711_scale)

        self.get_logger().info("Ejecutando tara inicial de balanza HX711 (peso de brazo = 0g)...")
        try:
            self.balanza.tare(15)
            self.get_logger().info(
                f"Tara completada. Offset={self.balanza.offset:.1f}, Escala={self.balanza.scale:.1f}"
            )
        except Exception as exc:
            self.get_logger().error(f"Fallo en tara inicial HX711: {exc}")

        # ── QoS, Suscripciones y Publicadores ──────────────────────────────
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

        self._pub_peso = self.create_publisher(
            Float32,
            self.TOPIC_PESO_ELEMENTO,
            10,
        )

        self._pub_estado = self.create_publisher(
            String,
            self.TOPIC_ESTADO_RECICLAJE,
            10,
        )

        # Publicacion continua de peso a 5 Hz cuando el sistema esta en reposo
        self._timer_peso = self.create_timer(0.2, self._publicar_peso_periodico)

        self.get_logger().info(
            f"NodoActuadores inicializado correctamente\n"
            f"  GPIO Chip        : gpiochip{self._gpio_chip}\n"
            f"  Motores          : M1(IN1={self.IN1}, IN2={self.IN2}), M2(IN3={self.IN3}, IN4={self.IN4})\n"
            f"  Sensores IR      : Izq=GPIO{self.SENSOR_IZQ}, Der=GPIO{self.SENSOR_DER} (PULL_UP)\n"
            f"  HX711 Balanza    : DT=GPIO{self.HX711_DT}, SCK=GPIO{self.HX711_SCK} (Escala: {self._hx711_scale})\n"
            f"  Max Lata         : {self._peso_max_lata}g\n"
            f"  Max Botella      : {self._peso_max_botella}g\n"
            f"  Timeout Ciclo    : {self._timeout_ciclo}s"
        )

    def _publicar_peso_periodico(self) -> None:
        """
        Publica periodicamente el peso actual en /peso_elemento cuando no hay ciclos de motor activos.
        """
        with self._lock:
            if self._hilo_ciclo is not None and self._hilo_ciclo.is_alive():
                return
        try:
            peso = self.balanza.get_units(3)
            msg = Float32()
            msg.data = float(peso)
            self._pub_peso.publish(msg)
        except Exception:
            pass

    def _init_gpio(self) -> int:
        """
        Reclama y configura los pines GPIO de motores y sensores IR.
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

        return handle

    def _cb_comando_grados(self, msg: Float32) -> None:
        """
        Procesa comandos por angulo.
        El ciclo cerrado maneja el recorrido completo de 360 grados de forma autonoma;
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
                    "Comando 0.0 ignorado: el ciclo realiza el recorrido completo 360 y centrado automaticamente."
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
        Maquina de estados para Sensor Fusion y ejecucion del ciclo de reciclaje continuo de 360 grados.
        """
        self.get_logger().info(f"Iniciando ciclo de reciclaje para accion: {accion}")

        # ────────────────────────────────────────────────────────────────────
        # FASE 0: VALIDACION DE PESO (Compuerta Logica de Seguridad)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 0: Validando peso del envase con celda de carga HX711...")
        try:
            peso_actual = self.balanza.get_units(5)
            self.get_logger().info(f"Lectura de peso actual: {peso_actual:.2f} g")
            self._pub_peso.publish(Float32(data=float(peso_actual)))
        except Exception as exc:
            self.get_logger().error(f"Fallo de lectura en sensor HX711: {exc}")
            self._pub_estado.publish(String(data="ERROR_BALANZA"))
            self._frenar_motores()
            return

        # Validacion de rangos especificos segun tipo de elemento (solo limite superior de proteccion)
        if accion == "GIRAR_LATA":
            peso_valido = peso_actual <= self._peso_max_lata
            limite_max = self._peso_max_lata
            direccion_ida = DIR_IZQUIERDA
        elif accion == "GIRAR_BOTELLA":
            peso_valido = peso_actual <= self._peso_max_botella
            limite_max = self._peso_max_botella
            direccion_ida = DIR_DERECHA
        else:
            self.get_logger().error(f"Accion desconocida: {accion}")
            return

        if not peso_valido:
            self.get_logger().error(
                f"Envase rechazado por exceso de peso. Retire el envase. "
                f"(Peso={peso_actual:.2f}g > Maximo={limite_max}g)"
            )
            self._pub_estado.publish(String(data="RECHAZADO_EXCESO_PESO"))
            self._frenar_motores()
            return

        self.get_logger().info(
            f"Fase 0 superada con exito (Peso={peso_actual:.2f}g <= Maximo={limite_max}g). Iniciando giro 360."
        )

        # ────────────────────────────────────────────────────────────────────
        # FASES 1-3: CICLO 360 CONTINUO UNIDIRECCIONAL
        # ────────────────────────────────────────────────────────────────────
        exito = self._ejecutar_ciclo_expulsion(direccion_ida)
        if exito:
            self.get_logger().info(f"Ciclo 360 de reciclaje ({accion}) completado exitosamente.")
            self._pub_estado.publish(String(data="RECICLAJE_EXITOSO"))
        else:
            self.get_logger().warning(f"Ciclo 360 de reciclaje ({accion}) finalizo con errores o timeout.")
            self._pub_estado.publish(String(data="ERROR_CICLO_ACTUADORES"))

    def _ejecutar_ciclo_expulsion(
        self, direccion_ida: Tuple[int, int, int, int]
    ) -> bool:
        """
        Ejecuta el ciclo continuo unidireccional de 360 grados:
            1. Fase de Escape: Giro continuo en direccion_ida hasta detectar 1-1 (salida del nido central).
            2. Fase de Trayecto: Giro continuo en la MISMA direccion (caida libre del envase) hasta detectar blanco.
            3. Fase de Anclaje: Bucle de auto-ajuste continuo hasta 0-0 con pulso de inercia y frenado.
        """
        t_inicio = time.time()
        timeout = self._timeout_ciclo
        pulso_inercia = self._pulso_inercia

        # ────────────────────────────────────────────────────────────────────
        # FASE 1: ESCAPE (Salir del nido central)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 1: Escape inicial del punto cero...")
        self._aplicar_estado_motores(*direccion_ida)

        # Esperar a que el brazo salga de la cinta blanca (ambos sensores en 1)
        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            if izq == 1 and der == 1:
                self.get_logger().info("Brazo fuera de rango central (1-1). Pasando a Fase 2 (Trayecto 360).")
                break

            time.sleep(0.001)
        else:
            self.get_logger().warning("Timeout en Fase 1 (Escape). Abortando ciclo.")
            self._frenar_motores()
            return False

        # ────────────────────────────────────────────────────────────────────
        # FASE 2: TRAYECTO (Giro continuo 360 y caida por gravedad)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 2: Trayecto continuo 360 en misma direccion...")
        # Mantener motores en direccion_ida sin invertir marcha
        self._aplicar_estado_motores(*direccion_ida)

        # Seguir girando continuamente hasta que cualquier sensor vuelva a detectar la cinta blanca (deje de ser 1-1)
        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            if izq == 0 or der == 0:
                self.get_logger().info(
                    f"Borde central detectado tras trayecto (Izq={izq}, Der={der}). Pasando a Fase 3."
                )
                break

            time.sleep(0.001)
        else:
            self.get_logger().warning("Timeout en Fase 2 (Trayecto). Abortando ciclo.")
            self._frenar_motores()
            return False

        # ────────────────────────────────────────────────────────────────────
        # FASE 3: ANCLAJE (Bucle de auto-ajuste y centrado continuo)
        # ────────────────────────────────────────────────────────────────────
        self.get_logger().info("Fase 3: Borde detectado. Iniciando auto-ajuste de anclaje continuo...")
        ultima_direccion_movimiento = direccion_ida

        while (time.time() - t_inicio) < timeout:
            if self._cancelar.is_set():
                self._frenar_motores()
                return False

            izq = lgpio.gpio_read(self._h, self.SENSOR_IZQ)
            der = lgpio.gpio_read(self._h, self.SENSOR_DER)

            # Estado 0-0: Centro perfecto alcanzado
            if izq == 0 and der == 0:
                self._aplicar_estado_motores(*ultima_direccion_movimiento)
                time.sleep(pulso_inercia)
                self._frenar_motores()
                self.get_logger().info(
                    "Centro perfecto alcanzado (0-0). Re-calibrando cero de la balanza..."
                )
                try:
                    self.balanza.tare(5)
                except Exception as exc:
                    self.get_logger().warning(f"Error en auto-tara dinamica de balanza: {exc}")
                return True

            elif izq == 0 and der == 1:
                # Tapo el izquierdo, falta el derecho -> Mover a la derecha
                ultima_direccion_movimiento = DIR_DERECHA
                self._aplicar_estado_motores(*DIR_DERECHA)

            elif izq == 1 and der == 0:
                # Tapo el derecho, falta el izquierdo -> Mover a la izquierda
                ultima_direccion_movimiento = DIR_IZQUIERDA
                self._aplicar_estado_motores(*DIR_IZQUIERDA)

            else:
                # 1-1: Por inercia se paso de largo y salio de la cinta
                # Seguir buscando en la direccion original del giro 360
                ultima_direccion_movimiento = direccion_ida
                self._aplicar_estado_motores(*direccion_ida)

            # Polling cooperativo de alta velocidad
            time.sleep(0.001)

        self.get_logger().warning("Timeout en Fase 3 (Anclaje). Abortando ciclo.")
        self._frenar_motores()
        return False

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
