#!/usr/bin/env python3
"""
nodo_actuadores.py
==================
Nodo ROS 2 (Jazzy Jalisco) que controla un motor paso a paso NEMA 17 mediante
el driver TB6600, usando lgpio sobre /dev/gpiochip4 (Raspberry Pi 5).

Tópicos de suscripción:
    /comando_motor  (std_msgs/Int32)   → Pasos directos (+CW / -CCW / 0=stop)
    /comando_grados (std_msgs/Float32) → Grados → convertidos internamente a pasos

Pinout TB6600 → Raspberry Pi 5 (BCM):
    PUL+  → GPIO 17   (pin_step)
    DIR+  → GPIO 27   (pin_dir)
    GND   → GND compartido con la RPi5

Hardware : Raspberry Pi 5 — Ubuntu 24.04
GPIO lib : lgpio  (sudo apt install python3-lgpio)

Conversión grados → pasos:
    pasos = int(round((grados / 360.0) * pasos_por_rev))
    Con pasos_por_rev=3200 (microstepping 1/16):
        90°  →  800 pasos
        45°  →  400 pasos
        1°   →    8.89 pasos ≈ 9 pasos

Velocidad:
    El parámetro 'delay_pulso' (s) controla el tiempo de cada semi-ciclo del
    pulso STEP. Puede cambiarse en caliente sin recompilar:
        ros2 param set /nodo_actuadores delay_pulso 0.001

    El nuevo valor se aplica al PRÓXIMO comando recibido.

NOTA sobre permisos en RPi5:
    Si lgpio lanza "can't open /dev/gpiochip4":
        sudo usermod -aG gpio $USER   (cierra sesión y vuelve a entrar)
    Para depurar sin reiniciar sesión:
        sudo ros2 run stepper_control_pkg nodo_actuadores
"""

import time
import threading

# ── Importación diferida de lgpio ──────────────────────────────────────────
# lgpio solo existe en Raspberry Pi con el paquete python3-lgpio instalado.
# Si falta (máquina de desarrollo, instalación incompleta) se captura aquí
# para dar un mensaje claro en lugar de un ImportError crudo.
try:
    import lgpio
    _LGPIO_DISPONIBLE = True
except ImportError:
    lgpio = None  # type: ignore[assignment]
    _LGPIO_DISPONIBLE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Int32, Float32


# ── Constantes de dirección ───────────────────────────────────────────────
_DIR_CW:  int = 1   # DIR+ HIGH → giro horario       (CW,  positivo)
_DIR_CCW: int = 0   # DIR+ LOW  → giro antihorario   (CCW, negativo)


class NodoActuadores(Node):
    """
    Nodo suscriptor de comandos de movimiento para el NEMA 17 / TB6600.

    Ambos tópicos convergen en el mismo hilo daemon de ejecución de pasos,
    garantizando que nunca haya dos movimientos simultáneos.

    Parámetros ROS 2:
        gpio_chip    (int)   : chip GPIO [default: 4  → /dev/gpiochip4 en RPi5]
        pin_step     (int)   : pin PUL+  [default: 17]
        pin_dir      (int)   : pin DIR+  [default: 27]
        pasos_por_rev(int)   : pasos/revolución incluyendo microstepping [default: 3200]
        delay_pulso  (float) : semi-ciclo del pulso STEP en segundos     [default: 0.0005]
                               → Periodo completo = 2 × delay_pulso
                               → 0.0005 s → 1 ms/ciclo → 1000 pasos/s → ≈18.75 RPM
    """

    TOPIC_COMANDO_MOTOR  = "/comando_motor"
    TOPIC_COMANDO_GRADOS = "/comando_grados"

    def __init__(self) -> None:
        super().__init__("nodo_actuadores")

        # ── Verificación de hardware GPIO ──────────────────────────────────
        if not _LGPIO_DISPONIBLE:
            self.get_logger().fatal(
                "lgpio no está instalado o no es importable. "
                "Ejecuta: sudo apt install python3-lgpio\n"
                "El nodo_actuadores no puede arrancar sin acceso GPIO."
            )
            raise RuntimeError("lgpio no disponible — abortando nodo_actuadores")

        # ── Parámetros ─────────────────────────────────────────────────────
        self.declare_parameter("gpio_chip",    4)
        self.declare_parameter("pin_step",    17)
        self.declare_parameter("pin_dir",     27)
        self.declare_parameter("pasos_por_rev", 3200)
        self.declare_parameter("delay_pulso",   0.0005)

        self._gpio_chip:     int = self.get_parameter("gpio_chip").get_parameter_value().integer_value
        self._pin_step:      int = self.get_parameter("pin_step").get_parameter_value().integer_value
        self._pin_dir:       int = self.get_parameter("pin_dir").get_parameter_value().integer_value
        # pasos_por_rev y delay_pulso se leen dinámicamente en cada movimiento
        # (ver _ejecutar_pasos) para respetar cambios hechos con ros2 param set.

        # ── Estado del hilo de movimiento ──────────────────────────────────
        # DISEÑO DE CONCURRENCIA:
        # self._lock   → protege SOLO la escritura/lectura de self._hilo_motor.
        #               NO se adquiere mientras se espera join() para evitar deadlock.
        # self._cancelar → threading.Event; señala al hilo daemon que pare.
        #
        # Flujo correcto para cancelar + relanzar:
        #   1. _cancelar.set()
        #   2. Leer referencia al hilo (con lock, muy rápido)
        #   3. Soltar lock
        #   4. join() SIN lock (puede tardar hasta delay_pulso × 2 por ciclo)
        #   5. Adquirir lock para escribir nuevo hilo
        self._hilo_motor: threading.Thread | None = None
        self._cancelar   = threading.Event()
        self._lock       = threading.Lock()

        # ── QoS ───────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Suscripciones ─────────────────────────────────────────────────
        self._sub_pasos = self.create_subscription(
            Int32,
            self.TOPIC_COMANDO_MOTOR,
            self._cb_comando_motor,
            qos,
        )

        self._sub_grados = self.create_subscription(
            Float32,
            self.TOPIC_COMANDO_GRADOS,
            self._cb_comando_grados,
            qos,
        )

        # ── Inicialización GPIO ────────────────────────────────────────────
        self._h: int = self._init_gpio()

        pasos_por_rev = self.get_parameter("pasos_por_rev").get_parameter_value().integer_value
        delay_pulso   = self.get_parameter("delay_pulso").get_parameter_value().double_value

        self.get_logger().info(
            f"NodoActuadores listo\n"
            f"  GPIO  : gpiochip{self._gpio_chip} | PUL=GPIO{self._pin_step} | DIR=GPIO{self._pin_dir}\n"
            f"  Motor : {pasos_por_rev} pasos/rev | delay={delay_pulso*1000:.2f} ms/semi-ciclo\n"
            f"  Topics: '{self.TOPIC_COMANDO_MOTOR}' (Int32 pasos directos)\n"
            f"          '{self.TOPIC_COMANDO_GRADOS}' (Float32 grados)"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Inicialización GPIO
    # ══════════════════════════════════════════════════════════════════════

    def _init_gpio(self) -> int:
        """
        Abre /dev/gpiochipN y reclama PUL y DIR como salidas digitales.
        Lanza excepción lgpio directamente si hay error de permisos o chip.
        """
        handle = lgpio.gpiochip_open(self._gpio_chip)
        lgpio.gpio_claim_output(handle, self._pin_step, 0)
        lgpio.gpio_claim_output(handle, self._pin_dir,  0)
        self.get_logger().info(
            f"GPIO OK → /dev/gpiochip{self._gpio_chip} | "
            f"PUL=GPIO{self._pin_step} DIR=GPIO{self._pin_dir}"
        )
        return handle

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks de suscripción
    # ══════════════════════════════════════════════════════════════════════

    def _cb_comando_motor(self, msg: Int32) -> None:
        """
        Recibe pasos directos (Int32) y lanza el movimiento.

        +N → N pasos CW
        -N → N pasos CCW
         0 → detener movimiento en curso
        """
        pasos = msg.data
        self.get_logger().info(
            f"[/comando_motor] {pasos:+d} pasos directos"
        )
        self._lanzar_movimiento(pasos)

    def _cb_comando_grados(self, msg: Float32) -> None:
        """
        Recibe un ángulo en grados (Float32), lo convierte a pasos y lanza
        el movimiento usando el mismo hilo daemon que /comando_motor.

        Conversión:
            pasos = int(round((grados / 360.0) * pasos_por_rev))

        Ejemplos con pasos_por_rev=3200:
            +90.0°  →  +800 pasos  (CW)
            -45.5°  →  -404 pasos  (CCW)
            +360.0° → +3200 pasos  (1 vuelta completa CW)
               0.0° →    0 pasos   (stop)
        """
        grados = msg.data
        pasos_por_rev: int = (
            self.get_parameter("pasos_por_rev").get_parameter_value().integer_value
        )

        pasos = int(round((grados / 360.0) * pasos_por_rev))

        self.get_logger().info(
            f"[/comando_grados] {grados:+.2f}° → {pasos:+d} pasos "
            f"(pasos_por_rev={pasos_por_rev})"
        )
        self._lanzar_movimiento(pasos)

    # ══════════════════════════════════════════════════════════════════════
    # Punto de entrada común para ambos tópicos
    # ══════════════════════════════════════════════════════════════════════

    def _lanzar_movimiento(self, pasos: int) -> None:
        """
        Cancela el movimiento en curso (si existe) y lanza uno nuevo en un
        hilo daemon separado.

        Garantías:
            - Nunca hay dos hilos de motor corriendo al mismo tiempo.
            - El hilo anterior siempre termina antes de empezar el nuevo.
            - Si pasos==0, solo cancela sin iniciar nada.

        CORRECCIÓN DE DEADLOCK (Bug #1):
            El lock NO se mantiene durante join() porque join() puede tardar
            hasta un semi-ciclo de pulso. Capturamos la referencia al hilo
            bajo el lock (operación O(1)), soltamos el lock, y luego hacemos
            join() sin el lock.
        """
        # Paso 1: señalar cancelación
        self._cancelar.set()

        # Paso 2: capturar referencia al hilo actual bajo el lock (muy rápido)
        with self._lock:
            hilo_anterior = self._hilo_motor

        # Paso 3: join() SIN lock — puede esperar hasta un ciclo de pulso (~1 ms)
        if hilo_anterior and hilo_anterior.is_alive():
            hilo_anterior.join(timeout=3.0)
            if hilo_anterior.is_alive():
                self.get_logger().warning(
                    "Advertencia: el hilo de motor no terminó en 3 s."
                )

        if pasos == 0:
            self.get_logger().info("Comando 0 recibido — motor detenido.")
            return

        # Paso 4: limpiar evento y lanzar nuevo hilo bajo el lock
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
    # Bucle de pulsos (hilo daemon)
    # ══════════════════════════════════════════════════════════════════════

    def _ejecutar_pasos(self, pasos: int) -> None:
        """
        Genera exactamente abs(pasos) pulsos STEP respetando la dirección.

        El parámetro 'delay_pulso' se lee al inicio de cada movimiento, de
        modo que un `ros2 param set /nodo_actuadores delay_pulso X` surte
        efecto en el próximo comando sin reiniciar el nodo.

        Estructura de cada pulso:
            PUL → HIGH  ──── delay_pulso ────  PUL → LOW  ──── delay_pulso ────
            |←──────────────── 1 ciclo completo ───────────────────────────────→|
        """
        n_pasos    = abs(pasos)
        direccion  = _DIR_CW if pasos > 0 else _DIR_CCW
        dir_str    = "CW (+)" if pasos > 0 else "CCW (-)"

        # Lee delay_pulso en el momento de inicio (recoge ros2 param set)
        delay_objetivo: float = (
            self.get_parameter("delay_pulso").get_parameter_value().double_value
        )
        
        # Parámetros de rampa
        delay_inicial = 0.005
        pasos_rampa = 800  # Pasos de rampa para 3200 ppr (suaviza la transición)
        if n_pasos < pasos_rampa * 2:
            pasos_rampa = n_pasos // 2

        periodo_ms = delay_objetivo * 2 * 1000   # periodo crucero en ms (para log)

        # ── Fija la dirección y da tiempo de setup al TB6600 ──────────────
        lgpio.gpio_write(self._h, self._pin_dir, direccion)
        time.sleep(0.005)   # 5 ms de settle para el TB6600 tras cambio de DIR

        self.get_logger().info(
            f"Movimiento iniciado | {dir_str} | {n_pasos} pasos | "
            f"delay_crucero={delay_objetivo*1000:.2f} ms | rampa={pasos_rampa} pasos"
        )

        pasos_hechos = 0

        for i in range(n_pasos):

            # Comprueba cancelación antes de cada pulso
            if self._cancelar.is_set():
                self.get_logger().info(
                    f"Movimiento CANCELADO en paso {pasos_hechos}/{n_pasos}"
                )
                break

            # Calcular delay actual (rampa trapezoidal)
            if i < pasos_rampa:
                # Aceleración
                progreso = i / pasos_rampa
                delay_actual = delay_inicial - progreso * (delay_inicial - delay_objetivo)
            elif i >= n_pasos - pasos_rampa:
                # Desaceleración
                progreso = (n_pasos - i) / pasos_rampa
                delay_actual = delay_inicial - progreso * (delay_inicial - delay_objetivo)
            else:
                # Velocidad crucero
                delay_actual = delay_objetivo

            # ── Pulso STEP ────────────────────────────────────────────────
            lgpio.gpio_write(self._h, self._pin_step, 1)   # flanco de subida
            time.sleep(delay_actual)                       # HIGH durante delay
            lgpio.gpio_write(self._h, self._pin_step, 0)   # flanco de bajada
            time.sleep(delay_actual)                       # LOW durante delay

            pasos_hechos += 1

        else:
            # Bucle completado sin cancelación
            self.get_logger().info(
                f"Movimiento COMPLETADO | {pasos_hechos}/{n_pasos} pasos | {dir_str}"
            )

        # Garantiza que PUL quede en LOW al terminar
        lgpio.gpio_write(self._h, self._pin_step, 0)

    # ══════════════════════════════════════════════════════════════════════
    # Control del hilo
    # ══════════════════════════════════════════════════════════════════════

    def _cancelar_movimiento_actual(self) -> None:
        """
        Señala cancelación y espera hasta 3 s a que el hilo de motor termine.
        Llamado desde destroy_node(), no desde _lanzar_movimiento().
        """
        self._cancelar.set()
        with self._lock:
            hilo = self._hilo_motor
        # join() fuera del lock para no bloquear otros threads que lean _lock
        if hilo and hilo.is_alive():
            hilo.join(timeout=3.0)
            if hilo.is_alive():
                self.get_logger().warning(
                    "Advertencia: el hilo de motor no terminó en 3 s."
                )

    # ══════════════════════════════════════════════════════════════════════
    # Ciclo de vida
    # ══════════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        """Detiene el motor, lleva los pines a LOW y cierra el chip GPIO."""
        self.get_logger().info("NodoActuadores: apagando — cancelando movimiento.")
        self._cancelar_movimiento_actual()
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
    try:
        nodo = NodoActuadores()
    except RuntimeError as exc:
        # RuntimeError lanzado si lgpio no está disponible — ya logueado en __init__
        rclpy.shutdown()
        return
    except Exception as exc:
        # Cualquier otro fallo de inicialización (permisos GPIO, chip no encontrado)
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
