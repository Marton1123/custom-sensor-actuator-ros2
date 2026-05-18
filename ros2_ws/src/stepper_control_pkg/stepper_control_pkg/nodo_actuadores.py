#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 (Jazzy Jalisco) que controla un motor paso a paso NEMA 17 mediante
el driver TB6600, usando lgpio para generar el tren de pulsos en la RPi5.

Suscripciones:
    /estado_sensor  (std_msgs/Float32) : 1.0 → girar | 0.0 → detener
    /control_manual (std_msgs/Bool)    : True → girar | False → detener

Pinout TB6600 → Raspberry Pi 5 (BCM, configurables por parámetros):
    PUL+ (STEP) → GPIO 17
    DIR+        → GPIO 27

Hardware target : Raspberry Pi 5 — Ubuntu 24.04
Librería GPIO   : lgpio  (sudo apt install python3-lgpio)

NOTA IMPORTANTE — gpiochip en RPi5:
    En Raspberry Pi 5 con Ubuntu 24.04, el GPIO principal está en
    /dev/gpiochip4, NO en gpiochip0 como en los modelos anteriores.
    El parámetro 'gpio_chip' (default=4) permite ajustarlo sin tocar el código.

Estrategia de generación de pulsos:
    Se usa lgpio.tx_pwm() — función no bloqueante que delega el tren de pulsos
    a un hilo interno de lgpio. Esto libera el hilo ROS 2 para seguir procesando
    mensajes mientras el motor gira de forma continua.

    Frecuencia de paso por defecto: 400 Hz → 2 rev/s con NEMA 17 full-step (200 p/rev).
    Ajusta 'step_freq' para cambiar la velocidad sin recompilar.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32, Bool

try:
    import lgpio
    _LGPIO_DISPONIBLE = True
except ImportError:
    _LGPIO_DISPONIBLE = False


# ── Valores de referencia del TB6600 ─────────────────────────────────────
_DIR_CW:  int = 1   # DIR+ HIGH → giro horario
_DIR_CCW: int = 0   # DIR+ LOW  → giro antihorario
_PWM_DUTY_CYCLE: float = 50.0   # 50 % — pulso cuadrado estándar para step drivers

# Umbral de comparación: si el valor Float32 >= este valor → girar
_UMBRAL_ACTIVO: float = 0.5


class NodoActuadores(Node):
    """
    Nodo suscriptor que controla un NEMA 17 via TB6600 con lgpio.

    Parámetros ROS 2:
        gpio_chip   (int)   : índice del chip GPIO (/dev/gpiochipN) [default: 4]
        pin_step    (int)   : pin STEP (PUL+) del TB6600            [default: 17]
        pin_dir     (int)   : pin DIR+  del TB6600                  [default: 27]
        step_freq   (float) : frecuencia del tren de pulsos (Hz)    [default: 400.0]
        dir_horario (bool)  : True=CW, False=CCW                    [default: True]
    """

    TOPIC_ESTADO_SENSOR  = "/estado_sensor"
    TOPIC_CONTROL_MANUAL = "/control_manual"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        # ── Parámetros ─────────────────────────────────────────────────────
        self.declare_parameter("gpio_chip",   4)
        self.declare_parameter("pin_step",    17)
        self.declare_parameter("pin_dir",     27)
        self.declare_parameter("step_freq",   400.0)
        self.declare_parameter("dir_horario", True)

        self._gpio_chip:   int   = self.get_parameter("gpio_chip").get_parameter_value().integer_value
        self._pin_step:    int   = self.get_parameter("pin_step").get_parameter_value().integer_value
        self._pin_dir:     int   = self.get_parameter("pin_dir").get_parameter_value().integer_value
        self._step_freq:   float = self.get_parameter("step_freq").get_parameter_value().double_value
        self._dir_horario: bool  = self.get_parameter("dir_horario").get_parameter_value().bool_value

        # ── Estado interno ─────────────────────────────────────────────────
        self._motor_girando: bool = False
        self._gpio_handle = None
        self._lock = threading.Lock()  # serializa acceso a GPIO desde callbacks

        # ── QoS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(  # Best-effort para sensor
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_rel = QoSProfile(  # Reliable para control manual
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Suscripciones ─────────────────────────────────────────────────
        self._sub_sensor = self.create_subscription(
            Float32,
            self.TOPIC_ESTADO_SENSOR,
            self._cb_estado_sensor,
            qos_be,
        )
        self._sub_manual = self.create_subscription(
            Bool,
            self.TOPIC_CONTROL_MANUAL,
            self._cb_control_manual,
            qos_rel,
        )

        # ── Inicialización GPIO ────────────────────────────────────────────
        self._init_gpio()

        self.get_logger().info(
            f"NodoActuadores iniciado | chip=gpiochip{self._gpio_chip} "
            f"STEP=GPIO{self._pin_step} DIR=GPIO{self._pin_dir} "
            f"freq={self._step_freq:.0f} Hz "
            f"| lgpio={'OK' if _LGPIO_DISPONIBLE else 'NO DISPONIBLE — modo log'}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Inicialización y teardown GPIO
    # ══════════════════════════════════════════════════════════════════════

    def _init_gpio(self) -> None:
        """
        Abre el chip GPIO y reclama los pines STEP y DIR como salidas.

        Si lgpio no está instalado, el nodo funciona en modo "solo log"
        para poder probar la lógica ROS 2 sin hardware.
        """
        if not _LGPIO_DISPONIBLE:
            self.get_logger().warning(
                "lgpio no disponible — ejecutando en MODO LOG (sin GPIO real). "
                "Instala con: sudo apt install python3-lgpio"
            )
            return

        try:
            # Abre /dev/gpiochipN (N=4 en RPi5 con Ubuntu 24.04)
            self._gpio_handle = lgpio.gpiochip_open(self._gpio_chip)

            # Reclama los pines como salidas digitales, nivel inicial LOW
            lgpio.gpio_claim_output(self._gpio_handle, self._pin_step, 0)
            lgpio.gpio_claim_output(self._gpio_handle, self._pin_dir,  0)

            # Establece la dirección de giro inicial
            dir_nivel = _DIR_CW if self._dir_horario else _DIR_CCW
            lgpio.gpio_write(self._gpio_handle, self._pin_dir, dir_nivel)

            self.get_logger().info(
                f"GPIO inicializado: chip=gpiochip{self._gpio_chip} | "
                f"STEP→GPIO{self._pin_step} | DIR→GPIO{self._pin_dir} "
                f"({'CW' if self._dir_horario else 'CCW'})"
            )

        except Exception as exc:
            self._gpio_handle = None
            self.get_logger().error(
                f"Error al inicializar GPIO: {exc}. "
                "Verifica que lgpio esté instalado y que tengas permisos "
                "(añade tu usuario al grupo 'gpio': sudo usermod -aG gpio $USER)."
            )

    def _cerrar_gpio(self) -> None:
        """Detiene el PWM y cierra el handle del chip GPIO."""
        if self._gpio_handle is None:
            return
        try:
            # Detiene cualquier tren de pulsos activo
            lgpio.tx_pwm(self._gpio_handle, self._pin_step, 0, 0)
            lgpio.gpiochip_close(self._gpio_handle)
            self._gpio_handle = None
            self.get_logger().info("GPIO liberado correctamente.")
        except Exception as exc:
            self.get_logger().error(f"Error al cerrar GPIO: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Control del motor
    # ══════════════════════════════════════════════════════════════════════

    def _iniciar_giro(self) -> None:
        """
        Inicia el tren de pulsos continuo en el pin STEP usando lgpio.tx_pwm().

        lgpio.tx_pwm(handle, gpio, frecuencia_Hz, ciclo_trabajo_%, offset=0, count=0)
            - count=0  → pulsos infinitos hasta llamar a _detener_giro()
            - ciclo    → 50 % genera pulso cuadrado simétrico

        Esta llamada es NO BLOQUEANTE: lgpio maneja el PWM en background,
        liberando el hilo ROS 2 para seguir procesando mensajes.
        """
        if self._gpio_handle is None:
            self.get_logger().info(
                f"[MODO LOG] Motor INICIADO — freq={self._step_freq:.0f} Hz "
                f"({'CW' if self._dir_horario else 'CCW'})"
            )
            return

        try:
            # Actualiza dirección antes de arrancar
            dir_nivel = _DIR_CW if self._dir_horario else _DIR_CCW
            lgpio.gpio_write(self._gpio_handle, self._pin_dir, dir_nivel)

            # Arranca el tren de pulsos (non-blocking)
            lgpio.tx_pwm(
                self._gpio_handle,
                self._pin_step,
                self._step_freq,    # Hz
                _PWM_DUTY_CYCLE,    # %
                0,                  # offset µs
                0,                  # count=0 → infinito
            )
            self.get_logger().info(
                f"Motor GIRANDO | freq={self._step_freq:.0f} Hz "
                f"≈ {self._step_freq / 200:.1f} rev/s (full-step, 200 p/rev)"
            )
        except Exception as exc:
            self.get_logger().error(f"Error al iniciar PWM: {exc}")

    def _detener_giro(self) -> None:
        """
        Detiene el tren de pulsos y deja el pin STEP en LOW.

        Pasando frecuencia=0 y duty=0 a tx_pwm se cancela la transacción activa.
        """
        if self._gpio_handle is None:
            self.get_logger().info("[MODO LOG] Motor DETENIDO")
            return

        try:
            # Frecuencia 0 cancela el PWM en lgpio
            lgpio.tx_pwm(self._gpio_handle, self._pin_step, 0, 0)
            # Asegura que el pin quede en LOW (el TB6600 no interpreta un nivel fijo como paso)
            lgpio.gpio_write(self._gpio_handle, self._pin_step, 0)
            self.get_logger().info("Motor DETENIDO | STEP pin → LOW")
        except Exception as exc:
            self.get_logger().error(f"Error al detener PWM: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks de suscripción
    # ══════════════════════════════════════════════════════════════════════

    def _cb_estado_sensor(self, msg: Float32) -> None:
        """
        Reacciona al valor publicado por nodo_sensores.

        valor >= 0.5  (≡ 1.0) → girar de forma continua
        valor <  0.5  (≡ 0.0) → detener el motor

        El _lock evita condiciones de carrera si /control_manual llega
        simultáneamente desde otro hilo del ejecutor.
        """
        valor = msg.data
        self.get_logger().debug(f"/estado_sensor recibido: {valor:.1f}")

        with self._lock:
            if valor >= _UMBRAL_ACTIVO:
                if not self._motor_girando:
                    self._motor_girando = True
                    self._iniciar_giro()
            else:
                if self._motor_girando:
                    self._motor_girando = False
                    self._detener_giro()

    def _cb_control_manual(self, msg: Bool) -> None:
        """
        Override manual: permite controlar el motor desde la terminal sin
        necesidad del tópico del sensor.

        Uso desde terminal:
            ros2 topic pub /control_manual std_msgs/msg/Bool "data: true"  --once
            ros2 topic pub /control_manual std_msgs/msg/Bool "data: false" --once
        """
        self.get_logger().info(f"/control_manual recibido: {msg.data}")
        with self._lock:
            if msg.data:
                if not self._motor_girando:
                    self._motor_girando = True
                    self._iniciar_giro()
            else:
                if self._motor_girando:
                    self._motor_girando = False
                    self._detener_giro()

    # ══════════════════════════════════════════════════════════════════════
    # Ciclo de vida
    # ══════════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        """Detiene el motor y libera GPIO antes de destruir el nodo."""
        self.get_logger().info("NodoActuadores: apagando nodo — deteniendo motor.")
        with self._lock:
            if self._motor_girando:
                self._detener_giro()
                self._motor_girando = False
        self._cerrar_gpio()
        super().destroy_node()


# ── Punto de entrada ──────────────────────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoActuadores()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
