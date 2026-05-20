#!/usr/bin/env python3
"""
nodo_balanza.py — Lector de celda de carga HX711 (Principio de Responsabilidad Única).

Este módulo provee la interfaz para interactuar exclusivamente con un
módulo HX711 conectado a una celda de carga. En caso de no poder inicializar
el hardware real (debido a incompatibilidad con el chip RP1 en Raspberry Pi 5),
hace uso de una clase Mock que simula los pesos.
"""

import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class MockHX711:
    """
    Simulación de celda de carga HX711.

    Genera valores aleatorios entre 10g y 150g para desarrollo y pruebas
    en entornos sin acceso directo a los pines GPIO o incompatibilidad
    con el chip RP1 de Raspberry Pi 5.
    """

    def __init__(self, dout_pin: int, sck_pin: int):
        """
        Inicializa la celda simulada.

        Args:
            dout_pin (int): Pin de datos (ignorado en mock).
            sck_pin (int): Pin de reloj (ignorado en mock).
        """
        self._dout = dout_pin
        self._sck = sck_pin

    def get_weight(self) -> float:
        """
        Devuelve un peso simulado en gramos.
        Simula la lectura diferencial del HX711 internamente.

        Returns:
            float: Peso simulado entre 10.0 y 150.0 gramos.
        """
        return random.uniform(10.0, 150.0)


class NodoBalanza(Node):
    """
    Nodo ROS 2 para la publicación del peso de la balanza.

    Se encarga de configurar el hardware (o mock) y de publicar el peso
    leído de la celda de carga a través de un tópico Float32 a una
    frecuencia de 10 Hz.
    """

    def __init__(self):
        """
        Inicializa el nodo ROS 2, el publicador del peso y el temporizador.
        """
        super().__init__('nodo_balanza')

        self._pub_peso = self.create_publisher(Float32, '/peso_botella', 10)

        # Hardware real vs Mock
        # La lectura del HX711 en la RPi5 puede ser problemática con bibliotecas viejas.
        # Aquí usamos directamente la simulación (Mock) para asegurar continuidad,
        # como indica el requerimiento. Se puede sustituir 'MockHX711' por la
        # clase de hardware real cuando haya una biblioteca compatible con RP1.
        try:
            self._hx711 = MockHX711(dout_pin=5, sck_pin=6)
            self.get_logger().info("Iniciada balanza en modo MOCK (10g - 150g).")
        except Exception as e:
            self.get_logger().warn(f"Fallo en hardware, usando mock: {e}")
            self._hx711 = MockHX711(dout_pin=5, sck_pin=6)

        # 10 Hz = 0.1 segundos
        self._timer = self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self) -> None:
        """
        Callback del temporizador que se ejecuta a 10 Hz.
        
        Lee el peso del dispositivo HX711 (o su mock) y lo publica en el
        tópico `/peso_botella`.
        """
        peso = self._hx711.get_weight()
        msg = Float32()
        msg.data = float(peso)
        self._pub_peso.publish(msg)
        self.get_logger().debug(f"Peso publicado: {peso:.2f} g")


def main(args=None):
    """
    Punto de entrada principal del nodo_balanza.
    """
    rclpy.init(args=args)
    nodo = NodoBalanza()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.get_logger().info("Nodo detenido manualmente.")
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
