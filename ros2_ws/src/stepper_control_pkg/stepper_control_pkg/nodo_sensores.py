#!/usr/bin/env python3
"""
nodo_sensores.py
================
Nodo ROS 2 (Jazzy Jalisco) que simula un sensor publicando datos dummy en el
tópico /estado_sensor (std_msgs/Float32).

Comportamiento del dato dummy:
    • Publica 1.0 durante FASE_ALTA_S segundos.
    • Publica 0.0 durante FASE_BAJA_S segundos.
    • Ciclo infinito.

Hardware target : Raspberry Pi 5 — Ubuntu 24.04
Dependencias    : ros-jazzy-std-msgs, python3-rclpy

Cuando dispongas del sensor físico, reemplaza _read_sensor() con la lectura real
y elimina las referencias a _inicio_fase / _en_fase_alta.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32


# ── Duración de cada fase del ciclo dummy ─────────────────────────────────
FASE_ALTA_S: float = 5.0   # segundos en que se publica 1.0
FASE_BAJA_S: float = 5.0   # segundos en que se publica 0.0


class NodoSensores(Node):
    """
    Nodo publicador de /estado_sensor.

    Parámetros ROS 2:
        sample_rate  (int)   : Frecuencia de publicación en Hz  [default: 10]
    """

    TOPIC_ESTADO_SENSOR = "/estado_sensor"

    def __init__(self) -> None:
        super().__init__("nodo_sensores")

        # ── Parámetros ─────────────────────────────────────────────────────
        self.declare_parameter("sample_rate", 10)  # Hz
        self._sample_rate: int = (
            self.get_parameter("sample_rate").get_parameter_value().integer_value
        )

        # ── Estado del ciclo dummy ─────────────────────────────────────────
        self._inicio_fase: float = time.monotonic()
        self._en_fase_alta: bool = True   # empieza publicando 1.0

        # ── QoS ───────────────────────────────────────────────────────────
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

        # ── Timer ─────────────────────────────────────────────────────────
        periodo_s: float = 1.0 / self._sample_rate
        self._timer = self.create_timer(periodo_s, self._timer_callback)

        self.get_logger().info(
            f"NodoSensores iniciado | tópico='{self.TOPIC_ESTADO_SENSOR}' "
            f"| {self._sample_rate} Hz | ciclo dummy {FASE_ALTA_S}s×1.0 / "
            f"{FASE_BAJA_S}s×0.0"
        )

    # ------------------------------------------------------------------
    # Lectura del sensor (dummy: ciclo 1.0/0.0)
    # ------------------------------------------------------------------
    def _read_sensor(self) -> float:
        """
        Devuelve el valor del sensor.

        Lógica dummy:
            - Primeros FASE_ALTA_S segundos de la fase: retorna 1.0
            - Segundos FASE_BAJA_S segundos de la fase: retorna 0.0
            - Luego vuelve a 1.0, y así en ciclo infinito.

        Para conectar el sensor real: elimina toda la lógica de fase y
        retorna directamente la lectura del hardware, p. ej.:
            return float(self._canal_adc.voltage)
        """
        ahora = time.monotonic()
        tiempo_en_fase = ahora - self._inicio_fase

        if self._en_fase_alta:
            if tiempo_en_fase >= FASE_ALTA_S:
                # Cambia a fase baja
                self._en_fase_alta = False
                self._inicio_fase = ahora
                self.get_logger().info("Ciclo dummy: cambiando a FASE BAJA (0.0)")
            return 1.0
        else:
            if tiempo_en_fase >= FASE_BAJA_S:
                # Cambia a fase alta
                self._en_fase_alta = True
                self._inicio_fase = ahora
                self.get_logger().info("Ciclo dummy: cambiando a FASE ALTA (1.0)")
            return 0.0

    # ------------------------------------------------------------------
    # Callback del timer
    # ------------------------------------------------------------------
    def _timer_callback(self) -> None:
        """Publica el valor del sensor en /estado_sensor cada 1/sample_rate s."""
        valor = self._read_sensor()

        msg = Float32()
        msg.data = valor
        self._publisher.publish(msg)

        self.get_logger().debug(f"Publicando /estado_sensor: {valor:.1f}")

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        self.get_logger().info("NodoSensores: apagando nodo.")
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
