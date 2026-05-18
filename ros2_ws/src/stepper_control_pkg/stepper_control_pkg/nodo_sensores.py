#!/usr/bin/env python3
"""
nodo_sensores.py
================
Nodo ROS 2 (Jazzy Jalisco) que lee datos de un sensor analógico/digital
customizado y publica el estado en el tópico /estado_sensor.

Hardware target : Raspberry Pi 5 — Ubuntu 24.04
Driver de sensor : a definir (ADS1115, MCP3008, GPIO directo, etc.)

Dependencias del sistema (instalar en la RPi5):
    sudo apt install python3-rclpy ros-jazzy-std-msgs
    pip3 install gpiozero  # o lgpio / RPi.GPIO según el driver elegido
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Mensaje estándar para publicar el estado del sensor.
# Cambia a un msg customizado si necesitas más campos (ej. Float32MultiArray).
from std_msgs.msg import Float32

# ---------------------------------------------------------------------------
# TODO (GPIO / sensor): importar la librería de bajo nivel para tu sensor
# Ejemplo con gpiozero + ADS1115 (I²C):
#   from gpiozero import MCP3008
# Ejemplo con Adafruit CircuitPython ADS1x15:
#   import board, busio, adafruit_ads1x15.ads1115 as ADS
#   from adafruit_ads1x15.analog_in import AnalogIn
# ---------------------------------------------------------------------------


class NodoSensores(Node):
    """
    Nodo publicador que lee un sensor y publica el valor en /estado_sensor.

    Parámetros ROS 2 (declarados en __init__):
        sample_rate  (int)   : Frecuencia de muestreo en Hz  [default: 10]
        sensor_pin   (int)   : Pin BCM o canal analógico      [default: 0]
    """

    TOPIC_ESTADO_SENSOR = "/estado_sensor"

    def __init__(self) -> None:
        super().__init__("nodo_sensores")

        # ── Parámetros configurables desde la línea de comandos / launch ──
        self.declare_parameter("sample_rate", 10)   # Hz
        self.declare_parameter("sensor_pin", 0)     # Pin BCM o canal ADC

        self._sample_rate: int = (
            self.get_parameter("sample_rate").get_parameter_value().integer_value
        )
        self._sensor_pin: int = (
            self.get_parameter("sensor_pin").get_parameter_value().integer_value
        )

        # ── QoS: Best-effort para datos de sensor en tiempo real ──────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self._publisher = self.create_publisher(
            Float32,
            self.TOPIC_ESTADO_SENSOR,
            qos,
        )

        # ── Timer de muestreo ─────────────────────────────────────────────
        timer_period_s: float = 1.0 / self._sample_rate
        self._timer = self.create_timer(timer_period_s, self._timer_callback)

        # ── Inicialización del hardware del sensor ────────────────────────
        self._init_sensor()

        self.get_logger().info(
            f"NodoSensores iniciado | tópico='{self.TOPIC_ESTADO_SENSOR}' "
            f"| {self._sample_rate} Hz | pin/canal={self._sensor_pin}"
        )

    # ------------------------------------------------------------------
    # Inicialización de hardware
    # ------------------------------------------------------------------
    def _init_sensor(self) -> None:
        """
        Configura el hardware del sensor.

        TODO (GPIO): Aquí va la inicialización de tu sensor, por ejemplo:

            # Con gpiozero + MCP3008 (SPI):
            self._sensor = MCP3008(channel=self._sensor_pin)

            # Con ADS1115 (I²C):
            i2c = busio.I2C(board.SCL, board.SDA)
            ads  = ADS.ADS1115(i2c)
            self._channel = AnalogIn(ads, ADS.P0)

            # Sensor digital (GPIO puro):
            self._gpio_pin = DigitalInputDevice(self._sensor_pin, pull_up=True)
        """
        self._sensor = None  # eliminar cuando se implemente el driver real
        self.get_logger().warning(
            "Hardware del sensor AÚN NO inicializado — modo simulación activo."
        )

    # ------------------------------------------------------------------
    # Lectura del sensor
    # ------------------------------------------------------------------
    def _read_sensor(self) -> float:
        """
        Lee el valor crudo del sensor y lo retorna como float.

        TODO (GPIO): Reemplaza el valor simulado con la lectura real:

            # MCP3008 / gpiozero:
            return float(self._sensor.value)         # 0.0 – 1.0

            # ADS1115:
            return self._channel.voltage             # voltaje en V

            # GPIO digital:
            return 1.0 if self._gpio_pin.is_active else 0.0
        """
        import math, time  # solo para simulación; eliminar después
        return round(math.sin(time.time()) * 50.0 + 50.0, 3)  # valor simulado

    # ------------------------------------------------------------------
    # Callback del timer
    # ------------------------------------------------------------------
    def _timer_callback(self) -> None:
        """Llamado cada 1/sample_rate segundos; publica la lectura del sensor."""
        valor = self._read_sensor()

        msg = Float32()
        msg.data = valor

        self._publisher.publish(msg)
        self.get_logger().debug(f"Publicando estado_sensor: {valor:.3f}")

    # ------------------------------------------------------------------
    # Limpieza al apagar el nodo
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        """
        TODO (GPIO): Libera recursos del sensor antes de destruir el nodo.

            self._sensor.close()   # gpiozero
        """
        self.get_logger().info("NodoSensores: limpiando recursos.")
        super().destroy_node()


# ── Punto de entrada ──────────────────────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoSensores()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
