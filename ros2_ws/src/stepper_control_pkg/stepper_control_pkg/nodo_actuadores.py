#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 que controla dos motores DC simultáneamente mediante el driver L298N (Puente H),
usando lgpio sobre /dev/gpiochip4 (Raspberry Pi 5).

Tópicos de suscripción:
    /comando_grados (std_msgs/Float32) → Lógica de control (90.0 para dirección Botella, -90.0 para dirección Lata)
    /objeto_detectado (std_msgs/String) → Lógica de IA ("bottle" para dirección Botella, "can" para dirección Lata)

Conexiones L298N → Raspberry Pi 5 (BCM):
    Motor 1 (Motor A): IN1 → GPIO 18, IN2 → GPIO 23
    Motor 2 (Motor B): IN3 → GPIO 25, IN4 → GPIO 26
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
    Nodo suscriptor de comandos de movimiento simultáneo para dos Motores DC / L298N.
    Se activa por tiempo (segundos) y frena ambos automáticamente.
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
        # Motor 1 (Motor A)
        self.IN1 = 18
        self.IN2 = 23
        # Motor 2 (Motor B)
        self.IN3 = 25
        self.IN4 = 26

        # Configuración de tiempo de funcionamiento simultáneo (en segundos)
        self.declare_parameter("tiempo_motor", 4.7)

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
            f"  GPIO     : gpiochip{self._gpio_chip}\n"
            f"  Motor 1  : IN1=GPIO{self.IN1} | IN2=GPIO{self.IN2}\n"
            f"  Motor 2  : IN3=GPIO{self.IN3} | IN4=GPIO{self.IN4}\n"
            f"  Tiempo   : {self.get_parameter('tiempo_motor').value} s\n"
            f"  Topics   : '{self.TOPIC_COMANDO_GRADOS}' y '/objeto_detectado'"
        )

    def _init_gpio(self) -> int:
        handle = lgpio.gpiochip_open(self._gpio_chip)
        lgpio.gpio_claim_output(handle, self.IN1, 0)
        lgpio.gpio_claim_output(handle, self.IN2, 0)
        lgpio.gpio_claim_output(handle, self.IN3, 0)
        lgpio.gpio_claim_output(handle, self.IN4, 0)
        self.get_logger().info(
            f"GPIO OK → /dev/gpiochip{self._gpio_chip} | "
            f"Motor 1: IN1=GPIO{self.IN1} IN2=GPIO{self.IN2} | "
            f"Motor 2: IN3=GPIO{self.IN3} IN4=GPIO{self.IN4}"
        )
        return handle

    def _cb_comando_grados(self, msg: Float32) -> None:
        """
        Lógica de activación por comandos en grados.
        """
        comando = msg.data
        self.get_logger().info(f"[/comando_grados] Recibido: {comando}")
        
        if comando == 90.0:
            self._lanzar_movimiento("GIRAR_BOTELLA")
        elif comando == -90.0:
            self._lanzar_movimiento("GIRAR_LATA")
        elif comando == 0.0:
            self.get_logger().info("Reset/Centro recibido (se ignora en control simple por tiempo).")
        else:
            self.get_logger().warning(f"Comando desconocido: {comando}")

    def _cb_objeto_detectado(self, msg: String) -> None:
        """
        Callback de IA por si recibe el comando directamente de la cámara.
        """
        objeto = msg.data.lower()
        if not objeto:
            return
            
        if objeto == "bottle":
            self.get_logger().info("[IA] BOTELLA detectada -> Activando ambos motores (Dirección 1).")
            self._lanzar_movimiento("GIRAR_BOTELLA")
        elif objeto == "can":
            self.get_logger().info("[IA] LATA detectada -> Activando ambos motores (Dirección 2).")
            self._lanzar_movimiento("GIRAR_LATA")

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
                name=f"motores_{accion}",
            )
            self._hilo_motor.start()

    def _ejecutar_accion(self, accion: str) -> None:
        """
        Ejecuta el giro de ambos motores de forma simultánea.
        """
        tiempo = self.get_parameter('tiempo_motor').value

        if accion == "GIRAR_BOTELLA":
            self.get_logger().info(f"Girando ambos motores a dirección BOTELLA (Dirección 1) por {tiempo}s...")
            self._activar_motores(1, 0, 1, 0, tiempo)
            self.get_logger().info("Giro completado.")
        elif accion == "GIRAR_LATA":
            self.get_logger().info(f"Girando ambos motores a dirección LATA (Dirección 2) por {tiempo}s...")
            self._activar_motores(0, 1, 0, 1, tiempo)
            self.get_logger().info("Giro completado.")

    def _activar_motores(self, m1_in1: int, m1_in2: int, m2_in3: int, m2_in4: int, duracion: float) -> None:
        """
        Activa los pines de ambos motores simultáneamente, espera la duración y frena ambos.
        """
        # Encender Motor 1 y Motor 2
        lgpio.gpio_write(self._h, self.IN1, m1_in1)
        lgpio.gpio_write(self._h, self.IN2, m1_in2)
        lgpio.gpio_write(self._h, self.IN3, m2_in3)
        lgpio.gpio_write(self._h, self.IN4, m2_in4)
        
        # Espera activa para permitir cancelación
        inicio = time.time()
        while (time.time() - inicio) < duracion:
            if self._cancelar.is_set():
                self.get_logger().info("Movimiento cancelado prematuramente.")
                break
            time.sleep(0.01)
            
        # Frenar ambos motores (todos los pines a 0)
        lgpio.gpio_write(self._h, self.IN1, 0)
        lgpio.gpio_write(self._h, self.IN2, 0)
        lgpio.gpio_write(self._h, self.IN3, 0)
        lgpio.gpio_write(self._h, self.IN4, 0)
        self.get_logger().info("Motores frenados.")

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
        """Detiene los motores, libera recursos y cierra pines."""
        self.get_logger().info("NodoActuadores: apagando — liberando recursos.")
        self._cancelar_movimiento_actual()
        try:
            # Poner todos los pines a 0
            lgpio.gpio_write(self._h, self.IN1, 0)
            lgpio.gpio_write(self._h, self.IN2, 0)
            lgpio.gpio_write(self._h, self.IN3, 0)
            lgpio.gpio_write(self._h, self.IN4, 0)
            
            # Liberar gpios y cerrar chip
            lgpio.gpio_free(self._h, self.IN1)
            lgpio.gpio_free(self._h, self.IN2)
            lgpio.gpio_free(self._h, self.IN3)
            lgpio.gpio_free(self._h, self.IN4)
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
