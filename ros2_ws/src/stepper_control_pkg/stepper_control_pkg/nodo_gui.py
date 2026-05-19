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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
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
        self._fn_estado = None
        self._fn_lock = threading.Lock()

        self._sub_cam = self.create_subscription(
            Image, self.TOPIC_CAM, self._cb_camara, QOS_VIDEO
        )

        self._sub_analisis = self.create_subscription(
            String, "/analisis_botella", self._cb_analisis, 10
        )

        self.get_logger().info(
            f"NodoGUI listo | publica '{self.TOPIC_CMD}' | "
            f"suscrito a '{self.TOPIC_CAM}' y '/analisis_botella'"
        )

    def _cb_analisis(self, msg: String) -> None:
        with self._fn_lock:
            fn = self._fn_estado
        if fn:
            fn(msg.data)

    def registrar_callback_estado(self, fn: callable) -> None:
        with self._fn_lock:
            self._fn_estado = fn

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
    senal_analisis: pyqtSignal = pyqtSignal(str)

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self.setWindowTitle("Panel de Control — NEMA 17")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)

        self._build_ui()

        # Conecta la senal al slot en el hilo Qt
        self.senal_frame_nuevo.connect(self._mostrar_frame)
        self.senal_analisis.connect(self._procesar_analisis)

        # Registra la emision de la senal como callback del nodo ROS.
        # DEBE ocurrir antes de hilo_ros.start() para evitar race condition.
        nodo.registrar_callback_frame(self.senal_frame_nuevo.emit)
        nodo.registrar_callback_estado(self.senal_analisis.emit)

        self._flujo_activo = False
        self._esperando_retiro = False
        self._tiempo_vacio_inicio = 0.0

        self.showFullScreen()

    # ── Construccion de UI ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        main_layout.addWidget(self._make_camera_label(), stretch=3)
        main_layout.addWidget(self._make_interactive_panel())
        main_layout.addWidget(self._make_display())
        main_layout.addWidget(self._make_control_group(), stretch=2)
        main_layout.addWidget(self._hline())

        fila_pie = QHBoxLayout()
        fila_pie.setSpacing(5)
        fila_pie.addWidget(self._make_stop_button())
        main_layout.addLayout(fila_pie)

    # ── Widgets ──────────────────────────────────────────────────────────

    def _make_camera_label(self) -> QLabel:
        lbl = QLabel()
        lbl.setObjectName("lbl_camara")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setText(
            "[ Esperando botella... ]"
        )
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumHeight(300)
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

    def _make_interactive_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self._lbl_interactivo = QLabel("Esperando botella...")
        self._lbl_interactivo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_interactivo.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background: #002200; padding: 10px; border: 2px solid #004400;")
        
        btn_layout = QHBoxLayout()
        self._btn_limpia = QPushButton("LIMPIA")
        self._btn_limpia.setStyleSheet("background-color: #00AA00; color: white; font-size: 24px; font-weight: bold; padding: 15px;")
        self._btn_limpia.clicked.connect(self._click_limpia)
        self._btn_limpia.hide()
        
        self._btn_sucia = QPushButton("SUCIA")
        self._btn_sucia.setStyleSheet("background-color: #AA0000; color: white; font-size: 24px; font-weight: bold; padding: 15px;")
        self._btn_sucia.clicked.connect(self._click_sucia)
        self._btn_sucia.hide()
        
        btn_layout.addWidget(self._btn_limpia)
        btn_layout.addWidget(self._btn_sucia)
        
        layout.addWidget(self._lbl_interactivo)
        layout.addLayout(btn_layout)
        return panel

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
        btn = QPushButton("[ PARO DE EMERGENCIA ]")
        btn.setObjectName("btn_stop")
        btn.setFixedHeight(90)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Envia 0 grados — detiene el motor inmediatamente")
        btn.clicked.connect(lambda: self._enviar(0.0))
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

    def _procesar_analisis(self, resultado_str: str) -> None:
        if self._esperando_retiro:
            if resultado_str == "vacio":
                if self._tiempo_vacio_inicio == 0.0:
                    self._tiempo_vacio_inicio = time.time()
                elif time.time() - self._tiempo_vacio_inicio >= 2.0:
                    # 2 segundos de vacio continuo -> Reset
                    self._esperando_retiro = False
                    self._flujo_activo = False
                    self._tiempo_vacio_inicio = 0.0
                    self._lbl_interactivo.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background: #002200; padding: 10px; border: 2px solid #004400;")
                    self._lbl_interactivo.setText("Esperando botella...")
            else:
                self._tiempo_vacio_inicio = 0.0
            return

        if not self._flujo_activo and resultado_str in ["grande", "chica"]:
            self._flujo_activo = True
            tamano = "Grande" if resultado_str == "grande" else "Chica"
            self._lbl_interactivo.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFAA00; background: #222200; padding: 10px; border: 2px solid #554400;")
            self._lbl_interactivo.setText(f"Botella {tamano} detectada. ¿Cuál es su estado?")
            self._btn_limpia.show()
            self._btn_sucia.show()

    def _click_limpia(self) -> None:
        self._btn_limpia.hide()
        self._btn_sucia.hide()
        self._lbl_interactivo.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background: #004400; padding: 10px; border: 2px solid #00AA00;")
        self._lbl_interactivo.setText("¡Reciclaje Exitoso!")
        QApplication.beep()
        self._enviar(90.0)
        self._esperando_retiro = True
        self._tiempo_vacio_inicio = 0.0

    def _click_sucia(self) -> None:
        self._btn_limpia.hide()
        self._btn_sucia.hide()
        self._lbl_interactivo.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF0000; background: #440000; padding: 10px; border: 2px solid #AA0000;")
        self._lbl_interactivo.setText("Botella sucia. Por favor, retírela de la máquina.")
        QApplication.beep()
        self._esperando_retiro = True
        self._tiempo_vacio_inicio = 0.0

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
