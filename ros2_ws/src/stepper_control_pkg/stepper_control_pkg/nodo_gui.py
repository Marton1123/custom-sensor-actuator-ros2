#!/usr/bin/env python3
"""
nodo_gui.py  —  Panel de Control Industrial Retro (PyQt6)
==========================================================
Modo Kiosco — pantalla vertical 480x800, sin bordes (FramelessWindowHint).
Estetica SCADA / LabVIEW clasico anos 90.

Arquitectura de hilos:
    Hilo principal → QApplication.exec()   (Qt event loop)
    Hilo daemon    → executor.spin()        (ROS 2: publisher + subscriber)

Topicos:
    Publica  → /comando_grados      (std_msgs/Float32)
    Suscribe → /camara/video_raw    (sensor_msgs/Image)  desde nodo_camara

Seguridad inter-hilo para la imagen:
    El callback ROS corre en el hilo daemon.
    Usamos pyqtSignal(QImage) para entregarla al hilo Qt de forma segura.
    PyQt6 encolara automaticamente la emision cuando detecte cruce de hilos.

Orden de inicializacion (critico para evitar race condition):
    1. rclpy.init() + NodoGUI()
    2. QApplication()
    3. VentanaPrincipal(nodo)  ← registra callback de frame
    4. executor.spin() en hilo daemon  ← arranca DESPUES del callback
    5. app.exec()  ← bloquea hilo principal
    6. executor.shutdown() + hilo.join()  ← para spin limpiamente
    7. nodo.destroy_node() + rclpy.shutdown()
"""

import subprocess
import sys
import threading
import time

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QSizePolicy,
    QMessageBox,
)


# ── QoS para video (debe coincidir con nodo_camara) ───────────────────────
QOS_VIDEO = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)


# ══════════════════════════════════════════════════════════════════════════
# Nodo ROS 2
# ══════════════════════════════════════════════════════════════════════════

class NodoGUI(Node):
    """
    Nodo ROS 2 del frontend:
      - Publica en /comando_grados (motor).
      - Se suscribe a /camara/video_raw y envia frames a la GUI via callback.
    """

    TOPIC_CMD = "/comando_grados"
    TOPIC_CAM = "/camara/video_raw"

    def __init__(self) -> None:
        super().__init__("nodo_gui")

        qos_cmd = QoSProfile(depth=10,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
        self._pub = self.create_publisher(Float32, self.TOPIC_CMD, qos_cmd)
        self._bridge = CvBridge()

        # Lock protege _fn_emit_frame entre hilo ROS y hilo Qt
        self._fn_emit_frame = None
        self._fn_lock = threading.Lock()

        self._sub_cam = self.create_subscription(
            Image, self.TOPIC_CAM, self._cb_camara, QOS_VIDEO
        )

        # Lógica Autónoma del Motor
        self._ultimo_tiempo_motor = 0.0
        self._sub_botella = self.create_subscription(
            Bool, "/botella_detectada", self._cb_botella_detectada, 10
        )

        self.get_logger().info(
            f"NodoGUI listo | publica '{self.TOPIC_CMD}' | "
            f"suscrito a '{self.TOPIC_CAM}' y '/botella_detectada'"
        )

    def _cb_botella_detectada(self, msg: Bool) -> None:
        """Logica autonoma: si detecta botella (cooldown 5s), mueve el motor."""
        if msg.data:
            tiempo_actual = time.time()
            if tiempo_actual - self._ultimo_tiempo_motor > 5.0:
                self.get_logger().info("Botella detectada: publicando 90.0° automáticamente.")
                self.publicar_grados(90.0)
                self._ultimo_tiempo_motor = tiempo_actual

    def registrar_callback_frame(self, fn: callable) -> None:
        """Registra la funcion que recibe QImage. Llamar ANTES de spin."""
        with self._fn_lock:
            self._fn_emit_frame = fn

    def _cb_camara(self, msg: Image) -> None:
        """Convierte Image ROS a QImage y la emite via senal Qt (thread-safe)."""
        with self._fn_lock:
            fn = self._fn_emit_frame
        if fn is None:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            h, w, ch = frame.shape
            # .tobytes() crea copia propia: sin memory leak por acumulacion de frames
            img = QImage(frame.tobytes(), w, h, ch * w,
                         QImage.Format.Format_RGB888)
            fn(img)
        except Exception as exc:
            self.get_logger().warn(f"Error procesando frame camara: {exc}",
                                   once=True)

    def publicar_grados(self, grados: float) -> None:
        msg = Float32()
        msg.data = float(grados)
        self._pub.publish(msg)
        self.get_logger().info(f"CMD → {grados:+.1f}°")


# ══════════════════════════════════════════════════════════════════════════
# Stylesheet — Windows 95 / SCADA industrial retro
# ══════════════════════════════════════════════════════════════════════════

WIN95_GRAY   = "#D4D0C8"
WIN95_WHITE  = "#FFFFFF"
WIN95_LIGHT  = "#EBEBEB"
WIN95_SHADOW = "#808080"
WIN95_DARK   = "#404040"
LCD_BG       = "#001A00"
LCD_FG       = "#00FF41"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {WIN95_GRAY};
    color: #000000;
    font-family: "Arial", "MS Sans Serif", sans-serif;
}}
QPushButton {{
    background-color: {WIN95_GRAY};
    color: #000000;
    font-family: "Arial", sans-serif;
    font-size: 19px;
    font-weight: bold;
    border-top:    3px solid {WIN95_WHITE};
    border-left:   3px solid {WIN95_WHITE};
    border-bottom: 3px solid {WIN95_DARK};
    border-right:  3px solid {WIN95_DARK};
    padding: 18px 14px;
}}
QPushButton:hover {{ background-color: {WIN95_LIGHT}; }}
QPushButton:pressed {{
    border-top:    3px solid {WIN95_DARK};
    border-left:   3px solid {WIN95_DARK};
    border-bottom: 3px solid {WIN95_WHITE};
    border-right:  3px solid {WIN95_WHITE};
    padding-top: 20px; padding-left: 16px;
    padding-bottom: 16px; padding-right: 12px;
}}
QPushButton#btn_stop {{
    background-color: #AA0000;
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 3px;
    border-top:    4px solid #FF7777;
    border-left:   4px solid #FF7777;
    border-bottom: 4px solid #330000;
    border-right:  4px solid #330000;
    padding: 14px 8px;
}}
QPushButton#btn_stop:hover  {{ background-color: #CC0000; }}
QPushButton#btn_stop:pressed {{
    background-color: #880000;
    border-top: 4px solid #330000; border-left: 4px solid #330000;
    border-bottom: 4px solid #FF7777; border-right: 4px solid #FF7777;
}}
QPushButton#btn_apagar {{
    background-color: #2A1A00;
    color: #FFA040;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 1px;
    border-top:    3px solid #FF8C00;
    border-left:   3px solid #FF8C00;
    border-bottom: 3px solid #0D0800;
    border-right:  3px solid #0D0800;
    padding: 14px 8px;
}}
QPushButton#btn_apagar:hover  {{ background-color: #4A3000; }}
QPushButton#btn_apagar:pressed {{
    background-color: #1A0E00;
    border-top: 3px solid #0D0800; border-left: 3px solid #0D0800;
    border-bottom: 3px solid #FF8C00; border-right: 3px solid #FF8C00;
}}
QLabel#lbl_display {{
    background-color: {LCD_BG};
    color: {LCD_FG};
    font-family: "Courier New", monospace;
    font-size: 20px;
    font-weight: bold;
    border-top:    3px solid {WIN95_SHADOW};
    border-left:   3px solid {WIN95_SHADOW};
    border-bottom: 3px solid {WIN95_WHITE};
    border-right:  3px solid {WIN95_WHITE};
    padding: 8px 16px;
    letter-spacing: 2px;
}}
QLabel#lbl_camara {{
    background-color: #000000;
    color: #004400;
    font-family: "Courier New", monospace;
    font-size: 14px;
    font-weight: bold;
    border-top:    4px solid {WIN95_DARK};
    border-left:   4px solid {WIN95_DARK};
    border-bottom: 4px solid {WIN95_WHITE};
    border-right:  4px solid {WIN95_WHITE};
}}
QGroupBox {{
    font-weight: bold;
    font-size: 13px;
    color: #000000;
    border-top:    2px solid {WIN95_SHADOW};
    border-left:   2px solid {WIN95_SHADOW};
    border-bottom: 2px solid {WIN95_WHITE};
    border-right:  2px solid {WIN95_WHITE};
    margin-top: 14px; padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    background-color: {WIN95_GRAY};
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {WIN95_SHADOW};
}}
"""


# ══════════════════════════════════════════════════════════════════════════
# Ventana principal
# ══════════════════════════════════════════════════════════════════════════

class VentanaPrincipal(QMainWindow):
    """
    Panel de control industrial retro — Modo Kiosco vertical 480x800.

    Layout:
        [lbl_camara    stretch=3]   <- video en vivo desde nodo_camara
        [lbl_display   fixed 55px]  <- LCD ultimo comando
        [control_group stretch=2]   <- botones 2x2
        [hline]
        [btn_stop  stretch=2 | btn_apagar stretch=1]
    """

    # Senal definida a nivel de clase (requerido por PyQt6).
    # Emitida desde hilo ROS, recibida en hilo Qt (cross-thread encolado automatico).
    senal_frame_nuevo: pyqtSignal = pyqtSignal(QImage)

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self.setWindowTitle("Panel de Control — NEMA 17")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)

        self._build_ui()

        # Conecta la senal al slot en el hilo Qt
        self.senal_frame_nuevo.connect(self._mostrar_frame)

        # Registra la emision de la senal como callback del nodo ROS.
        # DEBE ocurrir antes de hilo_ros.start() para evitar race condition.
        nodo.registrar_callback_frame(self.senal_frame_nuevo.emit)

        self.showFullScreen()

    # ── Construccion de UI ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        main_layout.addWidget(self._make_camera_label(), stretch=3)
        main_layout.addWidget(self._make_display())
        main_layout.addWidget(self._make_control_group(), stretch=2)
        main_layout.addWidget(self._hline())

        fila_pie = QHBoxLayout()
        fila_pie.setSpacing(5)
        fila_pie.addWidget(self._make_stop_button(),   stretch=2)
        fila_pie.addWidget(self._make_apagar_button(), stretch=1)
        main_layout.addLayout(fila_pie)

    # ── Widgets ──────────────────────────────────────────────────────────

    def _make_camera_label(self) -> QLabel:
        lbl = QLabel()
        lbl.setObjectName("lbl_camara")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setText(
            "██  CAMARA DE DETECCION  ██\n\n"
            "[ Esperando senal de nodo_camara... ]\n\n"
            "/camara/video_raw"
        )
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumHeight(120)
        lbl.setMaximumHeight(350)
        self._lbl_camara = lbl
        return lbl

    def _make_display(self) -> QLabel:
        self._lbl_display = QLabel("ULTIMO COMANDO:  ---")
        self._lbl_display.setObjectName("lbl_display")
        self._lbl_display.setFixedHeight(55)
        self._lbl_display.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter
        )
        return self._lbl_display

    def _make_control_group(self) -> QGroupBox:
        group = QGroupBox(" CONTROL DE MOVIMIENTO ")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group.setMaximumHeight(250)
        grid = QGridLayout(group)
        grid.setSpacing(10)
        grid.setContentsMargins(10, 14, 10, 10)

        botones = [
            ("+90°",               90.0,  0, 0),
            ("-90°",              -90.0,  0, 1),
            ("+1 VUELTA (+360°)", 360.0,  1, 0),
            ("-1 VUELTA (-360°)",-360.0,  1, 1),
        ]
        for texto, grados, fila, col in botones:
            btn = QPushButton(texto)
            btn.setMinimumSize(QSize(160, 90))
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, g=grados: self._enviar(g))
            grid.addWidget(btn, fila, col)
        return group

    def _make_stop_button(self) -> QPushButton:
        btn = QPushButton("■■  PARO DE EMERGENCIA  ■■")
        btn.setObjectName("btn_stop")
        btn.setFixedHeight(90)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Envia 0 grados — detiene el motor inmediatamente")
        btn.clicked.connect(lambda: self._enviar(0.0))
        return btn

    def _make_apagar_button(self) -> QPushButton:
        btn = QPushButton("🔌 APAGAR\nSISTEMA")
        btn.setObjectName("btn_apagar")
        btn.setFixedHeight(90)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Apaga la Raspberry Pi de forma segura")
        btn.clicked.connect(self._apagar_sistema)
        return btn

    # ── Slots y logica ───────────────────────────────────────────────────

    def _mostrar_frame(self, img: QImage) -> None:
        """
        Slot Qt — ejecutado en hilo principal al llegar un frame.
        setPixmap() descarta el pixmap anterior automaticamente: sin memory leak.
        """
        pixmap = QPixmap.fromImage(img)
        self._lbl_camara.setPixmap(
            pixmap.scaled(
                self._lbl_camara.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _enviar(self, grados: float) -> None:
        self._nodo.publicar_grados(grados)

        if grados == 0.0:
            texto = "ULTIMO COMANDO:  *** PARO ***"
            color = "#FF4444"
        else:
            signo = "+" if grados > 0 else ""
            texto = f"ULTIMO COMANDO:  {signo}{grados:.1f}°"
            color = "#00FF41" if grados > 0 else "#FFD700"

        self._lbl_display.setText(texto)
        self._lbl_display.setStyleSheet(
            f"background-color:{LCD_BG}; color:{color};"
            "font-family:'Courier New',monospace; font-size:20px; font-weight:bold;"
            f"border-top:3px solid {WIN95_SHADOW}; border-left:3px solid {WIN95_SHADOW};"
            f"border-bottom:3px solid {WIN95_WHITE}; border-right:3px solid {WIN95_WHITE};"
            "padding:8px 16px; letter-spacing:2px;"
        )

    def _apagar_sistema(self) -> None:
        """
        Apaga la RPi5 con confirmacion previa y feedback de error en pantalla.

        Requiere en /etc/sudoers (visudo):
            kiosco ALL=(ALL) NOPASSWD: /bin/systemctl poweroff
        """
        resp = QMessageBox.question(
            self,
            "Confirmar Apagado",
            "¿Seguro que deseas apagar el sistema?\n\nEsta accion es irreversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        self._nodo.get_logger().info("Apagado del sistema solicitado desde GUI.")
        try:
            resultado = subprocess.run(
                ["sudo", "systemctl", "poweroff"],
                capture_output=True, text=True, timeout=10,
            )
            if resultado.returncode != 0:
                self._nodo.get_logger().error(
                    f"poweroff fallo (returncode={resultado.returncode}): {resultado.stderr}"
                )
                QMessageBox.critical(
                    self, "Error de Apagado",
                    f"No se pudo apagar el sistema:\n{resultado.stderr}\n\n"
                    "Verifica sudoers NOPASSWD para 'systemctl poweroff'.",
                )
        except Exception as exc:
            self._nodo.get_logger().error(f"Excepcion en apagado: {exc}")
            QMessageBox.critical(
                self, "Error de Apagado",
                f"Excepcion al intentar apagar:\n{exc}",
            )

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFrameShadow(QFrame.Shadow.Sunken)
        return f

    def keyPressEvent(self, event) -> None:
        """Modo kiosco: ninguna tecla cierra la aplicacion."""
        super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    # 1. ROS 2
    rclpy.init(args=args)
    nodo = NodoGUI()

    # 2. Qt
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # 3. Ventana — registra callback ANTES de arrancar el hilo de spin
    ventana = VentanaPrincipal(nodo)

    # 4. Spin en hilo daemon con executor controlable para shutdown limpio
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(nodo)
    hilo_ros = threading.Thread(
        target=executor.spin,
        daemon=True,
        name="hilo_rclpy",
    )
    hilo_ros.start()

    # 5. Qt event loop — bloquea hasta que la app cierre
    exit_code = app.exec()

    # 6. Shutdown ordenado: señalar al executor, esperar al hilo
    executor.shutdown(timeout_sec=2.0)
    hilo_ros.join(timeout=3.0)

    # 7. Limpiar nodo y contexto ROS
    nodo.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
