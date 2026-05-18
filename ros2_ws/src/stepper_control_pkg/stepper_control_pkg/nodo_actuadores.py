#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 (Jazzy Jalisco) que controla un motor paso a paso NEMA 17 mediante
el driver TB6600, usando lgpio sobre /dev/gpiochip4 (Raspberry Pi 5).

Suscripción:
    /comando_motor  (std_msgs/Int32)
        • Valor positivo (+N) → DIR HIGH (CW)  + N pulsos en PUL
        • Valor negativo (-N) → DIR LOW  (CCW) + N pulsos en PUL
        • Valor 0             → cancela el movimiento en curso

Pinout TB6600 → Raspberry Pi 5 (BCM):
    PUL+  → GPIO 17   (pin_step)
    DIR+  → GPIO 27   (pin_dir)
    GND   → GND

Hardware : Raspberry Pi 5 — Ubuntu 24.04
GPIO lib : lgpio  (sudo apt install python3-lgpio)

NOTA sobre gpiochip en RPi5:
    La RPi5 expone sus GPIOs en /dev/gpiochip4.
    Si obtienes "permission denied", añade tu usuario al grupo gpio:
        sudo usermod -aG gpio $USER   (luego cierra sesión y vuelve a entrar)
    O ejecuta el nodo con sudo para depuración inicial.
"""

import time
import threading

# Import directo — sin fallback de simulación.
# Si lgpio no está instalado o no hay permisos, el proceso falla con el
# error real para que puedas depurarlo.
import lgpio

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Int32


# ── Constantes de hardware ────────────────────────────────────────────────
_DIR_CW:  int   = 1        # DIR+ HIGH → giro horario       (CW)
_DIR_CCW: int   = 0        # DIR+ LOW  → giro antihorario   (CCW)
_PULSE_DELAY_S: float = 0.002   # 2 ms por semi-ciclo del pulso STEP
                                 # → periodo completo = 4 ms → 250 pasos/s máx.
                                 # TB6600 datasheet: mín. PUL high = 5 µs,
                                 # así que 2 ms es muy conservador y seguro.


class NodoActuadores(Node):
    """
    Nodo suscriptor que ejecuta trenes de pulsos discretos sobre el TB6600.

    Cada mensaje en /comando_motor dispara un movimiento de N pasos en un
    hilo separado, de modo que el ejecutor ROS 2 nunca se bloquea.

    Parámetros ROS 2:
        gpio_chip    (int)   : índice del chip GPIO [default: 4  → /dev/gpiochip4]
        pin_step     (int)   : pin PUL+ del TB6600  [default: 17]
        pin_dir      (int)   : pin DIR+ del TB6600  [default: 27]
        pulse_delay  (float) : retardo (s) entre flanco HIGH y LOW del pulso
                               [default: 0.002]
    """

    TOPIC_COMANDO_MOTOR = "/comando_motor"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        # ── Parámetros ─────────────────────────────────────────────────────
        self.declare_parameter("gpio_chip",   4)
        self.declare_parameter("pin_step",   17)
        self.declare_parameter("pin_dir",    27)
        self.declare_parameter("pulse_delay", _PULSE_DELAY_S)

        self._gpio_chip:   int   = self.get_parameter("gpio_chip").get_parameter_value().integer_value
        self._pin_step:    int   = self.get_parameter("pin_step").get_parameter_value().integer_value
        self._pin_dir:     int   = self.get_parameter("pin_dir").get_parameter_value().integer_value
        self._pulse_delay: float = self.get_parameter("pulse_delay").get_parameter_value().double_value

        # ── Estado del hilo de movimiento ──────────────────────────────────
        self._hilo_motor: threading.Thread | None = None
        self._cancelar   = threading.Event()   # set() → aborta el bucle de pasos
        self._lock       = threading.Lock()    # serializa acceso al hilo

        # ── QoS ───────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Suscripción ───────────────────────────────────────────────────
        self._sub_cmd = self.create_subscription(
            Int32,
            self.TOPIC_COMANDO_MOTOR,
            self._cb_comando_motor,
            qos,
        )

        # ── Inicialización GPIO (sin simulación — falla si hay error) ──────
        self._h: int = self._init_gpio()

        self.get_logger().info(
            f"NodoActuadores listo | gpiochip{self._gpio_chip} | "
            f"PUL=GPIO{self._pin_step} | DIR=GPIO{self._pin_dir} | "
            f"delay_pulso={self._pulse_delay*1000:.1f} ms | "
            f"escuchando '{self.TOPIC_COMANDO_MOTOR}' (Int32)"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Inicialización GPIO
    # ══════════════════════════════════════════════════════════════════════

    def _init_gpio(self) -> int:
        """
        Abre /dev/gpiochip4 y reclama los pines PUL y DIR como salidas.

        Lanza excepción directamente si:
            - lgpio no puede abrir el chip (permisos, chip incorrecto)
            - el pin ya está reclamado por otro proceso

        Retorna el handle lgpio (int) para usar en el resto del nodo.
        """
        handle = lgpio.gpiochip_open(self._gpio_chip)

        # Reclama PUL+ como salida, estado inicial LOW
        lgpio.gpio_claim_output(handle, self._pin_step, 0)

        # Reclama DIR+ como salida, estado inicial LOW (CCW)
        lgpio.gpio_claim_output(handle, self._pin_dir, 0)

        self.get_logger().info(
            f"GPIO OK → /dev/gpiochip{self._gpio_chip} | "
            f"PUL=GPIO{self._pin_step} DIR=GPIO{self._pin_dir} reclamados como salida"
        )
        return handle

    # ══════════════════════════════════════════════════════════════════════
    # Callback del tópico /comando_motor
    # ══════════════════════════════════════════════════════════════════════

    def _cb_comando_motor(self, msg: Int32) -> None:
        """
        Recibe un entero e inicia el movimiento correspondiente.

        Flujo:
            1. Si hay un movimiento en curso, lo cancela y espera a que termine.
            2. Si el comando es 0, solo cancela (sin nuevo movimiento).
            3. Lanza un hilo daemon que ejecuta el tren de pulsos.
        """
        pasos = msg.data
        self.get_logger().info(
            f"Comando recibido: {pasos:+d} pasos "
            f"({'CW' if pasos > 0 else 'CCW' if pasos < 0 else 'STOP'})"
        )

        # Cancela el movimiento activo (si lo hay)
        self._cancelar_movimiento_actual()

        if pasos == 0:
            return  # solo era un stop

        # Limpia el flag de cancelación y lanza el nuevo hilo
        with self._lock:
            self._cancelar.clear()
            self._hilo_motor = threading.Thread(
                target=self._ejecutar_pasos,
                args=(pasos,),
                daemon=True,
                name=f"motor_{'cw' if pasos > 0 else 'ccw'}_{abs(pasos)}",
            )
            self._hilo_motor.start()

    # ══════════════════════════════════════════════════════════════════════
    # Lógica de movimiento (hilo separado)
    # ══════════════════════════════════════════════════════════════════════

    def _ejecutar_pasos(self, pasos: int) -> None:
        """
        Genera exactamente abs(pasos) pulsos en el pin PUL.

        Cada pulso:
            1. PUL → HIGH
            2. time.sleep(pulse_delay)   ← semiciclo HIGH
            3. PUL → LOW
            4. time.sleep(pulse_delay)   ← semiciclo LOW

        La dirección se establece una sola vez antes del bucle y no cambia
        durante el movimiento (requisito del TB6600).

        El bucle se puede interrumpir en cualquier momento poniendo
        self._cancelar para que el sistema responda a nuevos comandos.
        """
        n_pasos    = abs(pasos)
        direccion  = _DIR_CW if pasos > 0 else _DIR_CCW
        dir_str    = "CW  (+)" if pasos > 0 else "CCW (-)"

        # ── Establece la dirección antes de los pulsos ─────────────────
        lgpio.gpio_write(self._h, self._pin_dir, direccion)
        # Pequeño settle para que el TB6600 reconozca el cambio de DIR
        time.sleep(0.005)

        self.get_logger().info(
            f"Iniciando {n_pasos} pasos | DIR={dir_str} | "
            f"delay={self._pulse_delay*1000:.1f} ms/semiciclo | "
            f"tiempo estimado={n_pasos * self._pulse_delay * 2:.2f} s"
        )

        pasos_completados = 0

        for _ in range(n_pasos):

            # ── Comprueba si se pidió cancelar ────────────────────────
            if self._cancelar.is_set():
                self.get_logger().info(
                    f"Movimiento CANCELADO en paso {pasos_completados}/{n_pasos}"
                )
                break

            # ── Pulso STEP ────────────────────────────────────────────
            lgpio.gpio_write(self._h, self._pin_step, 1)   # flanco de subida
            time.sleep(self._pulse_delay)                   # HIGH durante delay
            lgpio.gpio_write(self._h, self._pin_step, 0)   # flanco de bajada
            time.sleep(self._pulse_delay)                   # LOW durante delay

            pasos_completados += 1

        else:
            # El bucle for llegó hasta el final sin cancelación
            self.get_logger().info(
                f"Movimiento COMPLETADO: {pasos_completados}/{n_pasos} pasos | "
                f"DIR={dir_str}"
            )

        # Garantiza que PUL quede en LOW al terminar (sea por fin o cancelación)
        lgpio.gpio_write(self._h, self._pin_step, 0)

    # ══════════════════════════════════════════════════════════════════════
    # Helpers de control de hilo
    # ══════════════════════════════════════════════════════════════════════

    def _cancelar_movimiento_actual(self) -> None:
        """
        Señala cancelación y espera hasta 3 s a que el hilo de motor termine.

        timeout=3 s cubre el peor caso: pulse_delay=0.002 s × ~1500 pasos
        antes de que el flag sea revisado.
        """
        self._cancelar.set()
        with self._lock:
            if self._hilo_motor and self._hilo_motor.is_alive():
                self._hilo_motor.join(timeout=3.0)
                if self._hilo_motor.is_alive():
                    self.get_logger().warning(
                        "El hilo de motor no terminó en 3 s — "
                        "puede haber un problema de bloqueo."
                    )

    # ══════════════════════════════════════════════════════════════════════
    # Ciclo de vida
    # ══════════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        """Cancela el movimiento activo, lleva los pines a LOW y cierra el chip."""
        self.get_logger().info("NodoActuadores: apagando — deteniendo motor.")

        self._cancelar_movimiento_actual()

        # Deja ambos pines en LOW antes de cerrar
        try:
            lgpio.gpio_write(self._h, self._pin_step, 0)
            lgpio.gpio_write(self._h, self._pin_dir,  0)
            lgpio.gpiochip_close(self._h)
            self.get_logger().info("GPIO liberado correctamente.")
        except Exception as exc:
            self.get_logger().error(f"Error al cerrar GPIO: {exc}")

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
