#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 (Jazzy Jalisco) que se suscribe a /estado_sensor (y opcionalmente
a /control_manual) y acciona un motor paso a paso NEMA 17 mediante un driver
TB6600 usando los pines GPIO de la Raspberry Pi 5.

Hardware target : Raspberry Pi 5 — Ubuntu 24.04
Driver          : TB6600 (Step / Direction / Enable)

Pinout TB6600 → RPi5 (BCM por defecto, ajustable por parámetros):
    STEP  → GPIO 17  (BCM)
    DIR   → GPIO 27  (BCM)
    ENA   → GPIO 22  (BCM)  [LOW = habilitado en TB6600]

Dependencias del sistema:
    sudo apt install python3-rclpy ros-jazzy-std-msgs
    pip3 install lgpio   # librería recomendada para RPi5 (reemplaza RPi.GPIO)
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32, Bool  # Bool para /control_manual (ejemplo)

# ---------------------------------------------------------------------------
# TODO (GPIO): importar la librería de bajo nivel para los GPIO
# Opción recomendada en RPi5 con Ubuntu 24.04:
#   import lgpio
# Alternativa con gpiozero (backend lgpio):
#   from gpiozero import OutputDevice
# ---------------------------------------------------------------------------


# ── Constantes del driver TB6600 ──────────────────────────────────────────
_STEP_PULSE_WIDTH_S: float = 2e-6   # 2 µs mínimo según datasheet TB6600
_ENA_SETTLE_S: float = 0.005        # 5 ms de espera tras habilitar el driver


class NodoActuadores(Node):
    """
    Nodo suscriptor que controla un motor NEMA 17 a través del driver TB6600.

    Suscripciones:
        /estado_sensor   (std_msgs/Float32) : valor del sensor que dispara el movimiento
        /control_manual  (std_msgs/Bool)    : override manual (True = paso adelante)

    Parámetros ROS 2:
        pin_step     (int)   : pin STEP del TB6600   [default: 17]
        pin_dir      (int)   : pin DIR del TB6600    [default: 27]
        pin_ena      (int)   : pin ENA del TB6600    [default: 22]
        steps_rev    (int)   : pasos por revolución  [default: 200]
        step_delay   (float) : retardo entre pasos (s) [default: 0.002]
        umbral       (float) : umbral del sensor para activar el motor [default: 50.0]
    """

    TOPIC_ESTADO_SENSOR = "/estado_sensor"
    TOPIC_CONTROL_MANUAL = "/control_manual"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        # ── Parámetros configurables ───────────────────────────────────────
        self.declare_parameter("pin_step",   17)
        self.declare_parameter("pin_dir",    27)
        self.declare_parameter("pin_ena",    22)
        self.declare_parameter("steps_rev",  200)
        self.declare_parameter("step_delay", 0.002)
        self.declare_parameter("umbral",     50.0)

        self._pin_step:   int   = self.get_parameter("pin_step").get_parameter_value().integer_value
        self._pin_dir:    int   = self.get_parameter("pin_dir").get_parameter_value().integer_value
        self._pin_ena:    int   = self.get_parameter("pin_ena").get_parameter_value().integer_value
        self._steps_rev:  int   = self.get_parameter("steps_rev").get_parameter_value().integer_value
        self._step_delay: float = self.get_parameter("step_delay").get_parameter_value().double_value
        self._umbral:     float = self.get_parameter("umbral").get_parameter_value().double_value

        # ── Estado interno ─────────────────────────────────────────────────
        self._motor_activo: bool = False
        self._lock = threading.Lock()  # protege acceso a GPIO desde callbacks

        # ── QoS ───────────────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_control = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Suscripciones ─────────────────────────────────────────────────
        self._sub_sensor = self.create_subscription(
            Float32,
            self.TOPIC_ESTADO_SENSOR,
            self._cb_estado_sensor,
            qos_sensor,
        )

        self._sub_manual = self.create_subscription(
            Bool,
            self.TOPIC_CONTROL_MANUAL,
            self._cb_control_manual,
            qos_control,
        )

        # ── Inicialización del hardware GPIO ───────────────────────────────
        self._init_gpio()

        self.get_logger().info(
            f"NodoActuadores iniciado | STEP={self._pin_step} DIR={self._pin_dir} "
            f"ENA={self._pin_ena} | umbral={self._umbral}"
        )

    # ------------------------------------------------------------------
    # Inicialización GPIO
    # ------------------------------------------------------------------
    def _init_gpio(self) -> None:
        """
        Configura los pines GPIO para el driver TB6600.

        TODO (GPIO): Reemplaza el bloque de simulación con código real.

        — Usando lgpio (recomendado para RPi5):

            self._h = lgpio.gpiochip_open(0)          # abre /dev/gpiochip0
            lgpio.gpio_claim_output(self._h, self._pin_step)
            lgpio.gpio_claim_output(self._h, self._pin_dir)
            lgpio.gpio_claim_output(self._h, self._pin_ena)
            lgpio.gpio_write(self._h, self._pin_ena, 1)  # deshabilitado al inicio
            lgpio.gpio_write(self._h, self._pin_dir, 0)  # dirección CW

        — Usando gpiozero (backend lgpio):

            from gpiozero import OutputDevice
            self._step_pin = OutputDevice(self._pin_step, active_high=True, initial_value=False)
            self._dir_pin  = OutputDevice(self._pin_dir,  active_high=True, initial_value=False)
            self._ena_pin  = OutputDevice(self._pin_ena,  active_high=False, initial_value=True)
        """
        self._gpio_handle = None  # eliminar cuando se implemente el driver real
        self.get_logger().warning(
            "GPIO AÚN NO inicializado — modo simulación activo (sin movimiento real)."
        )

    # ------------------------------------------------------------------
    # Control del motor: habilitar / deshabilitar
    # ------------------------------------------------------------------
    def _habilitar_motor(self, habilitar: bool) -> None:
        """
        Activa o desactiva la salida del driver TB6600 mediante el pin ENA.

        En TB6600: ENA en LOW → driver habilitado / ENA en HIGH → deshabilitado.

        TODO (GPIO):
            nivel = 0 if habilitar else 1
            lgpio.gpio_write(self._h, self._pin_ena, nivel)  # lgpio
            # O:
            self._ena_pin.on() if habilitar else self._ena_pin.off()  # gpiozero
        """
        estado = "HABILITADO" if habilitar else "DESHABILITADO"
        self.get_logger().info(f"Motor TB6600: {estado} (simulado)")
        if habilitar:
            time.sleep(_ENA_SETTLE_S)

    # ------------------------------------------------------------------
    # Control del motor: dirección
    # ------------------------------------------------------------------
    def _set_direccion(self, horario: bool) -> None:
        """
        Establece la dirección de giro.

        horario=True  → CW (DIR = HIGH)
        horario=False → CCW (DIR = LOW)

        TODO (GPIO):
            lgpio.gpio_write(self._h, self._pin_dir, 1 if horario else 0)  # lgpio
            # O:
            self._dir_pin.on() if horario else self._dir_pin.off()         # gpiozero
        """
        dir_str = "CW (horario)" if horario else "CCW (antihorario)"
        self.get_logger().debug(f"Dirección: {dir_str} (simulado)")

    # ------------------------------------------------------------------
    # Control del motor: enviar pasos
    # ------------------------------------------------------------------
    def _mover_pasos(self, n_pasos: int, horario: bool = True) -> None:
        """
        Envía `n_pasos` pulsos al driver TB6600.

        Cada pulso consiste en:
            1. STEP → HIGH
            2. Espera `_STEP_PULSE_WIDTH_S`
            3. STEP → LOW
            4. Espera `self._step_delay` (controla la velocidad)

        TODO (GPIO): Reemplaza el bloque de simulación con código real:

            for _ in range(n_pasos):
                lgpio.gpio_write(self._h, self._pin_step, 1)
                time.sleep(_STEP_PULSE_WIDTH_S)
                lgpio.gpio_write(self._h, self._pin_step, 0)
                time.sleep(self._step_delay - _STEP_PULSE_WIDTH_S)
        """
        self.get_logger().info(
            f"Moviendo {n_pasos} pasos {'CW' if horario else 'CCW'} "
            f"a {self._step_delay*1000:.1f} ms/paso [SIMULADO]"
        )
        # -- simulación de tiempo de movimiento --
        tiempo_total = n_pasos * self._step_delay
        time.sleep(min(tiempo_total, 0.5))  # máximo 500 ms en simulación

    # ------------------------------------------------------------------
    # Callback: /estado_sensor
    # ------------------------------------------------------------------
    def _cb_estado_sensor(self, msg: Float32) -> None:
        """
        Lógica de control automático basada en el valor del sensor.

        Si el valor supera el umbral → activa el motor CW.
        Si está por debajo        → detiene / desactiva el motor.
        """
        valor = msg.data
        self.get_logger().debug(f"estado_sensor recibido: {valor:.3f}")

        with self._lock:
            if valor >= self._umbral:
                if not self._motor_activo:
                    self._motor_activo = True
                    self._habilitar_motor(True)
                    self._set_direccion(horario=True)
                    self.get_logger().info(
                        f"Umbral superado ({valor:.2f} >= {self._umbral}): "
                        "motor ACTIVADO."
                    )

                # TODO: definir cuántos pasos dar por ciclo de control
                pasos_por_ciclo = 10
                self._mover_pasos(pasos_por_ciclo, horario=True)

            else:
                if self._motor_activo:
                    self._motor_activo = False
                    self._habilitar_motor(False)
                    self.get_logger().info(
                        f"Valor bajo umbral ({valor:.2f} < {self._umbral}): "
                        "motor DETENIDO."
                    )

    # ------------------------------------------------------------------
    # Callback: /control_manual
    # ------------------------------------------------------------------
    def _cb_control_manual(self, msg: Bool) -> None:
        """
        Override manual del motor.

        True  → un paso en dirección CW.
        False → deshabilita el motor.

        TODO: Ampliar con velocidad y dirección si se usa un msg más rico.
        """
        self.get_logger().info(f"Control manual recibido: {msg.data}")
        with self._lock:
            if msg.data:
                self._habilitar_motor(True)
                self._set_direccion(horario=True)
                self._mover_pasos(1, horario=True)
            else:
                self._habilitar_motor(False)
                self._motor_activo = False

    # ------------------------------------------------------------------
    # Limpieza al apagar el nodo
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        """
        Asegura que el motor quede deshabilitado y los GPIO liberados.

        TODO (GPIO):
            self._habilitar_motor(False)
            lgpio.gpiochip_close(self._h)   # lgpio
            # O:
            self._step_pin.close()          # gpiozero
            self._dir_pin.close()
            self._ena_pin.close()
        """
        self.get_logger().info(
            "NodoActuadores: deshabilitando motor y liberando GPIO."
        )
        self._habilitar_motor(False)
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
