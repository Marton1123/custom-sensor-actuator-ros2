#!/usr/bin/env python3
"""
nodo_balanza.py — Lector de celda de carga HX711 con Hardware Real.

Implementa la lectura del sensor HX711 utilizando gpiozero para máxima 
compatibilidad con el chip RP1 de Raspberry Pi 5.
Aplica tara automática al inicio y publica el peso ajustado.
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from gpiozero import DigitalOutputDevice, DigitalInputDevice


class HX711_GPIOZero:
    """
    Controlador para el sensor HX711 utilizando gpiozero.
    Garantiza compatibilidad con Raspberry Pi 5 (RP1) evitando RPi.GPIO.
    """
    def __init__(self, dout_pin: int, sck_pin: int, gain: int = 128):
        self._dout = DigitalInputDevice(dout_pin)
        self._sck = DigitalOutputDevice(sck_pin, initial_value=False)
        self._gain = gain
        self._offset = 0.0
        
        # Pulsos adicionales según la ganancia
        # 1 pulso extra = Gain 128, Channel A
        if self._gain == 128:
            self._extra_pulses = 1
        elif self._gain == 64:
            self._extra_pulses = 3
        elif self._gain == 32:
            self._extra_pulses = 2
        else:
            self._extra_pulses = 1

    def is_ready(self) -> bool:
        """
        Verifica si el sensor está listo para entregar datos.
        El sensor está listo cuando DOUT se pone a 0 (False).
        """
        return not self._dout.value

    def read_raw(self) -> int:
        """
        Lee los 24 bits crudos del sensor mediante bit-banging en gpiozero.
        
        Returns:
            int: Valor crudo leído con signo (Complemento a 2).
        """
        # Esperar a que el HX711 esté listo (timeout de 0.5s de seguridad)
        t0 = time.time()
        while not self.is_ready():
            if time.time() - t0 > 0.5:
                raise RuntimeError("Timeout esperando al HX711")
            time.sleep(0.001)

        value = 0
        # Leer los 24 bits (MSB a LSB)
        for _ in range(24):
            self._sck.on()
            self._sck.off()
            # Desplazar y añadir el bit leído
            value = (value << 1) | int(self._dout.value)

        # Pulsos adicionales para configurar la ganancia de la siguiente lectura
        for _ in range(self._extra_pulses):
            self._sck.on()
            self._sck.off()

        # Convertir complemento a 2 (24 bits) a un entero con signo en Python
        if value & 0x800000:
            value -= 0x1000000

        return value

    def read_average(self, times: int = 3) -> float:
        """
        Realiza múltiples lecturas y devuelve el promedio crudo.
        
        Args:
            times (int): Cantidad de muestras a tomar.
            
        Returns:
            float: Promedio de las lecturas.
        """
        suma = 0.0
        exitos = 0
        for _ in range(times):
            try:
                suma += self.read_raw()
                exitos += 1
            except RuntimeError:
                pass
        
        if exitos == 0:
            raise RuntimeError("Fallo repetido al leer el sensor HX711")
        
        return suma / exitos

    def set_offset(self, offset: float) -> None:
        """Establece el offset para hacer la tara."""
        self._offset = offset

    def get_offset(self) -> float:
        """Obtiene el offset actual."""
        return self._offset


class NodoBalanza(Node):
    """
    Nodo ROS 2 para la publicación del peso real de la balanza.
    Incluye rutina de tara automática en la inicialización y factor
    de calibración interno.
    """

    def __init__(self):
        """
        Inicializa el nodo ROS 2, los pines, realiza la tara automática
        y configura el temporizador de publicación.
        """
        super().__init__('nodo_balanza')

        self._pub_peso = self.create_publisher(Float32, '/peso_botella', 10)
        
        # Factor de calibración de clase
        self._reference_unit = 1.0
        self._hx711 = None
        
        # Rutina de Inicialización (Hardware Real y Tara Automática)
        try:
            self.get_logger().info("Inicializando hardware HX711 en pines DT=5, SCK=6...")
            self._hx711 = HX711_GPIOZero(dout_pin=5, sck_pin=6)
            
            # Realizamos una Tara (Tare) para establecer el 0 inicial
            self.get_logger().info("Realizando tara automática...")
            offset_inicial = self._hx711.read_average(times=10)
            self._hx711.set_offset(offset_inicial)
            self.get_logger().info(f"Tara completada. Offset configurado: {offset_inicial:.1f}")
            
        except Exception as e:
            self.get_logger().error(f"Fallo al inicializar el hardware HX711: {e}")
            # El nodo sigue vivo, el timer intentará reportar el error suavemente

        # Ejecutamos la lectura y publicación a 10 Hz
        self._timer = self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self) -> None:
        """
        Callback ejecutado a 10 Hz.
        Calcula el peso a partir de la lectura, restando el offset
        y dividiendo por reference_unit, con gestión de excepciones.
        """
        if self._hx711 is None:
            return

        try:
            # Leer promedio de muestras para suavizar a 10Hz
            raw = self._hx711.read_average(times=2)
            offset = self._hx711.get_offset()
            
            # Aplicamos la fórmula exigida: (raw - offset) / reference_unit
            peso = (raw - offset) / self._reference_unit
            
            msg = Float32()
            msg.data = float(peso)
            self._pub_peso.publish(msg)
            
            self.get_logger().debug(f"Peso publicado: {peso:.2f}")
            
        except Exception as e:
            # Capturamos excepciones sin que el nodo haga 'crash'
            self.get_logger().warn(
                f"Error al leer la balanza HX711: {e}", 
                throttle_duration_sec=2.0
            )


def main(args=None):
    """
    Punto de entrada principal para lanzar el nodo.
    """
    rclpy.init(args=args)
    nodo = NodoBalanza()
    
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.get_logger().info("Nodo de balanza detenido manualmente.")
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
