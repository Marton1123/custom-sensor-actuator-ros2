#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 que controla un motor DC con caja reductora mediante
el driver L298N (Puente H), usando lgpio sobre /dev/gpiochip4 (Raspberry Pi 5).

Tópicos de suscripción:
    /comando_grados (std_msgs/Float32) → Lógica de estados (BOTELLA, LATA, CENTRO)
    /objeto_detectado (std_msgs/String) → Lógica de IA (bottle, can)

Conexiones L298N → Raspberry Pi 5 (BCM):
    IN1 → GPIO 18
    IN2 → GPIO 23

Hardware : Raspberry Pi 5 — Ubuntu 24.04
GPIO lib : lgpio  (sudo apt install python3-lgpio)
"""

import time
import threading

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


class NodoActuadores(Node):
    """
    Nodo suscriptor de comandos de movimiento para Motor DC / L298N.
    Mantiene estado y usa control por tiempo.
    """

    TOPIC_COMANDO_GRADOS = "/comando_grados"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        if not _LGPIO_DISPONIBLE:
            self.get_logger().fatal(
                "lgpio no está instalado o no es importable. "
                "El nodo_actuadores no puede arrancar sin acceso GPIO."
            )
            raise RuntimeError("lgpio no disponible — abortando nodo_actuadores")

        # ── Parámetros GPIO y Tiempos ──────────────────────────────────────
        self.declare_parameter("gpio_chip", 4)
        self._gpio_chip = self.get_parameter("gpio_chip").get_parameter_value().integer_value
        
        # Pines del L298N
        self.IN1 = 18
        self.IN2 = 23

        # Configuración de tiempo
        self.tiempo_90_grados = 0.4  # segundos

        # Estado del motor
        self.posicion_actual = "CENTRO"  # "CENTRO", "BOTELLA", "LATA"

        # ── Estado del hilo de movimiento ──────────────────────────────────
        self._hilo_motor: threading.Thread | None = None
        self._cancelar = threading.Event()
        self._lock = threading.Lock()

        # ── QoS ───────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Suscripciones ─────────────────────────────────────────────────
        self._sub_grados = self.create_subscription(
            Float32,
            self.TOPIC_COMANDO_GRADOS,
            self._cb_comando_grados,
            qos,
        )

        self._sub_vision = self.create_subscription(
            String,
            "/objeto_detectado",
            self._cb_objeto_detectado,
            qos,
        )

        # ── Inicialización GPIO ────────────────────────────────────────────
        self._h = self._init_gpio()

        self.get_logger().info(
            f"NodoActuadores listo\n"
            f"  GPIO   : gpiochip{self._gpio_chip} | IN1=GPIO{self.IN1} | IN2=GPIO{self.IN2}\n"
            f"  Tiempo : {self.tiempo_90_grados} s\n"
            f"  Topics : '{self.TOPIC_COMANDO_GRADOS}'"
        )

    def _init_gpio(self) -> int:
        """
        Abre /dev/gpiochipN y reclama IN1 y IN2 como salidas.
        Inician en 0 (LOW) para que el motor empiece frenado.
        """
        handle = lgpio.gpiochip_open(self._gpio_chip)
        lgpio.gpio_claim_output(handle, self.IN1, 0)
        lgpio.gpio_claim_output(handle, self.IN2, 0)
        self.get_logger().info(
            f"GPIO OK → /dev/gpiochip{self._gpio_chip} | "
            f"IN1=GPIO{self.IN1} IN2=GPIO{self.IN2}"
        )
        return handle

    def _cb_comando_grados(self, msg: Float32) -> None:
        """
        Lógica de control de estados en base al comando en grados.
        """
        comando = msg.data
        self.get_logger().info(f"[/comando_grados] Recibido: {comando}")
        
        accion = ""
        if comando == 90.0:
            accion = "A_BOTELLA"
        elif comando == -90.0:
            accion = "A_LATA"
        elif comando == 0.0:
            accion = "RESET"
        else:
            self.get_logger().warning(f"Comando desconocido: {comando}")
            return

        self._lanzar_movimiento(accion)

    def _cb_objeto_detectado(self, msg: String) -> None:
        """
        Callback de IA por si recibe el comando directamente de la cámara.
        """
        objeto = msg.data.lower()
        if not objeto:
            return
            
        if objeto == "bottle":
            self.get_logger().info("[IA] Botella detectada.")
            self._lanzar_movimiento("A_BOTELLA")
        elif objeto == "can":
            self.get_logger().info("[IA] Lata detectada.")
            self._lanzar_movimiento("A_LATA")

    def _lanzar_movimiento(self, accion: str) -> None:
        """
        Cancela el movimiento en curso y lanza la acción en un hilo separado.
        """
        self._cancelar.set()

        with self._lock:
            hilo_anterior = self._hilo_motor

        if hilo_anterior and hilo_anterior.is_alive():
            hilo_anterior.join(timeout=3.0)

        with self._lock:
            self._cancelar.clear()
            self._hilo_motor = threading.Thread(
                target=self._ejecutar_accion,
                args=(accion,),
                daemon=True,
                name=f"motor_{accion}",
            )
            self._hilo_motor.start()

    def _ejecutar_accion(self, accion: str) -> None:
        """
        Ejecuta la acción solicitada respetando la máquina de estados.
        """
        if accion == "A_BOTELLA":
            if self.posicion_actual == "CENTRO":
                self.get_logger().info("Moviendo a BOTELLA (IN1=1, IN2=0)")
                self._activar_motor(1, 0, self.tiempo_90_grados)
                self.posicion_actual = "BOTELLA"
            else:
                self.get_logger().info(f"Ignorado. Estado actual: {self.posicion_actual}")

        elif accion == "A_LATA":
            if self.posicion_actual == "CENTRO":
                self.get_logger().info("Moviendo a LATA (IN1=0, IN2=1)")
                self._activar_motor(0, 1, self.tiempo_90_grados)
                self.posicion_actual = "LATA"
            else:
                self.get_logger().info(f"Ignorado. Estado actual: {self.posicion_actual}")

        elif accion == "RESET":
            if self.posicion_actual == "BOTELLA":
                self.get_logger().info("Retornando desde BOTELLA (IN1=0, IN2=1)")
                self._activar_motor(0, 1, self.tiempo_90_grados)
                self.posicion_actual = "CENTRO"
            elif self.posicion_actual == "LATA":
                self.get_logger().info("Retornando desde LATA (IN1=1, IN2=0)")
                self._activar_motor(1, 0, self.tiempo_90_grados)
                self.posicion_actual = "CENTRO"
            elif self.posicion_actual == "CENTRO":
                self.get_logger().info("Ya en CENTRO. No hace nada.")

    def _activar_motor(self, in1_val: int, in2_val: int, duracion: float) -> None:
        """
        Activa los pines del motor, espera el tiempo indicado (con checks de cancelación)
        y frena.
        """
        lgpio.gpio_write(self._h, self.IN1, in1_val)
        lgpio.gpio_write(self._h, self.IN2, in2_val)
        
        # Espera activa para permitir cancelación
        inicio = time.time()
        while (time.time() - inicio) < duracion:
            if self._cancelar.is_set():
                self.get_logger().info("Movimiento cancelado prematuramente.")
                break
            time.sleep(0.01)
            
        # Frenar (ambos a 0)
        lgpio.gpio_write(self._h, self.IN1, 0)
        lgpio.gpio_write(self._h, self.IN2, 0)
        self.get_logger().info("Motor frenado.")

    def _cancelar_movimiento_actual(self) -> None:
        """
        Señala cancelación y espera hasta 3 s a que el hilo de motor termine.
        Llamado desde destroy_node().
        """
        self._cancelar.set()
        with self._lock:
            hilo = self._hilo_motor
        if hilo and hilo.is_alive():
            hilo.join(timeout=3.0)

    def destroy_node(self) -> None:
        """Detiene el motor, libera recursos y cierra pines."""
        self.get_logger().info("NodoActuadores: apagando — liberando recursos.")
        self._cancelar_movimiento_actual()
        try:
            # Poner pines en 0
            lgpio.gpio_write(self._h, self.IN1, 0)
            lgpio.gpio_write(self._h, self.IN2, 0)
            
            # Liberar gpios y cerrar chip
            lgpio.gpio_free(self._h, self.IN1)
            lgpio.gpio_free(self._h, self.IN2)
            lgpio.gpiochip_close(self._h)
            self.get_logger().info("GPIO liberado correctamente.")
        except Exception as exc:
            self.get_logger().error(f"Error al cerrar GPIO: {exc}")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        nodo = NodoActuadores()
    except RuntimeError:
        # RuntimeError lanzado si lgpio no está disponible
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
