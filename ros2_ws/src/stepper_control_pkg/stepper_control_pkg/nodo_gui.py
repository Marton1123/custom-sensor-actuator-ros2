#!/usr/bin/env python3
"""
nodo_gui.py  —  Panel de Control Industrial Retro (PyQt6)
==========================================================
Modo Kiosco — pantalla vertical 480×800, sin bordes de ventana (FramelessWindowHint).
Estética SCADA / LabVIEW clásico años 90.

Arquitectura:
    Hilo principal → QApplication.exec()   (Qt event loop)
    Hilo daemon    → rclpy.spin(NodoGUI)    (ROS 2 background)

Publica en: /comando_grados (std_msgs/Float32)
"""

import sys
import threading

import cv2              # OpenCV — sudo apt install python3-opencv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32

from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QSizePolicy,
)


# ══════════════════════════════════════════════════════════════════════════
# Nodo ROS 2
# ══════════════════════════════════════════════════════════════════════════

class NodoGUI(Node):
    TOPIC = "/comando_grados"

    def __init__(self) -> None:
        super().__init__("nodo_gui")
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._pub = self.create_publisher(Float32, self.TOPIC, qos)
        self.get_logger().info(f"NodoGUI listo → '{self.TOPIC}'")

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
NAVY         = "#000080"
LCD_BG       = "#001A00"
LCD_FG       = "#00FF41"

STYLESHEET = f"""
/* ── Base ── */
QMainWindow, QWidget {{
    background-color: {WIN95_GRAY};
    color: #000000;
    font-family: "Arial", "MS Sans Serif", sans-serif;
}}

/* ── Botones de movimiento (efecto bevel 3D) ── */
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
QPushButton:hover {{
    background-color: {WIN95_LIGHT};
}}
QPushButton:pressed {{
    border-top:    3px solid {WIN95_DARK};
    border-left:   3px solid {WIN95_DARK};
    border-bottom: 3px solid {WIN95_WHITE};
    border-right:  3px solid {WIN95_WHITE};
    padding-top:   20px;
    padding-left:  16px;
    padding-bottom:16px;
    padding-right: 12px;
}}

/* ── Paro de emergencia ── */
QPushButton#btn_stop {{
    background-color: #AA0000;
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 3px;
    border-top:    4px solid #FF7777;
    border-left:   4px solid #FF7777;
    border-bottom: 4px solid #330000;
    border-right:  4px solid #330000;
    padding: 22px 14px;
}}
QPushButton#btn_stop:hover {{
    background-color: #CC0000;
}}
QPushButton#btn_stop:pressed {{
    background-color: #880000;
    border-top:    4px solid #330000;
    border-left:   4px solid #330000;
    border-bottom: 4px solid #FF7777;
    border-right:  4px solid #FF7777;
    padding-top:   24px;
    padding-left:  16px;
    padding-bottom:20px;
    padding-right: 12px;
}}

/* btn_salir eliminado — modo kiosco sin botón de cierre */

/* ── Barra de título (estilo Win95 activo) ── */
QLabel#lbl_titlebar {{
    background-color: {NAVY};
    color: #FFFFFF;
    font-size: 17px;
    font-weight: bold;
    font-family: "Arial", sans-serif;
    padding: 6px 14px;
    letter-spacing: 1px;
}}

/* ── Display LCD ── */
QLabel#lbl_display {{
    background-color: {LCD_BG};
    color: {LCD_FG};
    font-family: "Courier New", "Courier", monospace;
    font-size: 22px;
    font-weight: bold;
    border-top:    3px solid {WIN95_SHADOW};
    border-left:   3px solid {WIN95_SHADOW};
    border-bottom: 3px solid {WIN95_WHITE};
    border-right:  3px solid {WIN95_WHITE};
    padding: 10px 20px;
    letter-spacing: 2px;
}}

/* ── Estado ONLINE ── */
QLabel#lbl_status {{
    color: #006400;
    font-size: 13px;
    font-weight: bold;
    font-family: "Courier New", monospace;
    background: transparent;
}}

/* ── GroupBox clásico ── */
QGroupBox {{
    font-weight: bold;
    font-size: 13px;
    color: #000000;
    border-top:    2px solid {WIN95_SHADOW};
    border-left:   2px solid {WIN95_SHADOW};
    border-bottom: 2px solid {WIN95_WHITE};
    border-right:  2px solid {WIN95_WHITE};
    margin-top: 14px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    background-color: {WIN95_GRAY};
    color: #000000;
}}

/* ── Separadores ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {WIN95_SHADOW};
}}

/* ── Placeholder cámara ── */
QLabel#lbl_camara {{
    background-color: #000000;
    color: #004400;
    font-family: "Courier New", monospace;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 2px;
    /* Borde inset estilo 90s: sombra arriba/izq, claro abajo/der */
    border-top:    4px solid {WIN95_DARK};
    border-left:   4px solid {WIN95_DARK};
    border-bottom: 4px solid {WIN95_WHITE};
    border-right:  4px solid {WIN95_WHITE};
}}
"""


# ══════════════════════════════════════════════════════════════════════════
# Ventana principal
# ══════════════════════════════════════════════════════════════════════════

class VentanaPrincipal(QMainWindow):
    """
    Panel de control industrial retro — Modo Kiosco vertical 480×800.

    Layout (vertical):
        ┌──────────────────────────────┐
        │  PANEL DE CONTROL NEMA 17   │  ← Titlebar navy (sin botón cerrar)
        ├──────────────────────────────┤
        │  [  CÁMARA DE DETECCIÓN  ]  │  ← Placeholder cámara 480×270
        ├──────────────────────────────┤
        │  ÚLTIMO COMANDO: ---        │  ← Display LCD
        ├─ CONTROL DE MOVIMIENTO ─────┤
        │  [ +90° ]    [ -90°  ]      │
        │  [+360° ]    [ -360° ]      │
        ├──────────────────────────────┤
        │  ■■  PARO DE EMERGENCIA  ■■ │  ← Stop
        └──────────────────────────────┘
    """

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self.setWindowTitle("Panel de Control — NEMA 17")
        # ── MODO KIOSCO ──
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # WindowFullScreen garantiza pantalla completa en X11/Xorg (RPi5)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self._build_ui()          # crea self._lbl_camara (necesario antes del timer)
        self._init_camara()       # inicializa VideoCapture y QTimer
        self.showFullScreen()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)  # cero márgenes → pantalla completa
        main_layout.setSpacing(5)                    # espacio mínimo entre secciones

        # 1. Cámara — empieza en y=0, ocupa todo el espacio libre
        main_layout.addWidget(self._make_camera_placeholder(), stretch=3)

        # 2. Display LCD (ancho completo, sin márgenes)
        main_layout.addWidget(self._make_display())

        # 3. Botones de movimiento
        main_layout.addWidget(self._make_control_group(), stretch=2)

        # 4. Separador
        main_layout.addWidget(self._hline())

        # 5. PARO DE EMERGENCIA
        main_layout.addWidget(self._make_stop_button())

    # ── Placeholder cámara ──────────────────────────────────────────────

    def _make_camera_placeholder(self) -> QLabel:
        """
        Marco negro que ocupa todo el espacio disponible verticalmente.
        Expanding en ambos ejes → se estira para rellenar la pantalla.
        Cuando OpenCV esté disponible (Paso 3), se usará setPixmap() en lugar de setText().
        """
        lbl = QLabel()
        lbl.setObjectName("lbl_camara")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setText(
            "██  CÁMARA DE DETECCIÓN  ██\n\n"
            "[ SIN SEÑAL ]\n\n"
            "Ch.01  480×800  LIVE"
        )
        # Expanding en ambos ejes → rellena el espacio sobrante
        lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        lbl.setMinimumHeight(120)   # mínimo de seguridad
        lbl.setMaximumHeight(350)   # tope → evita que robe todo el alto en pantallas pequeñas
        self._lbl_camara = lbl      # referencia para actualizar desde el timer
        return lbl

    # ── Display LCD (ancho completo) ────────────────────────────────────

    def _make_display(self) -> QLabel:
        """Label tipo LCD de ancho completo que muestra el último comando enviado."""
        self._lbl_display = QLabel("ÚLTIMO COMANDO:  ---")
        self._lbl_display.setObjectName("lbl_display")
        self._lbl_display.setFixedHeight(60)
        self._lbl_display.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter
        )
        return self._lbl_display

    # ── GroupBox de control ───────────────────────────────────────────────

    def _make_control_group(self) -> QGroupBox:
        """Grid 2×2 de botones táctiles. Expanding vertical → crece con la pantalla."""
        group = QGroupBox(" CONTROL DE MOVIMIENTO ")
        group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        group.setMaximumHeight(250)  # tope → previene desbordamiento en 800px de alto
        grid = QGridLayout(group)
        grid.setSpacing(10)
        grid.setContentsMargins(10, 14, 10, 10)

        botones = [
            ("+90°",               90.0,   0, 0),
            ("-90°",              -90.0,   0, 1),
            ("+1 VUELTA (+360°)", 360.0,   1, 0),
            ("-1 VUELTA (-360°)",-360.0,   1, 1),
        ]

        for texto, grados, fila, col in botones:
            btn = QPushButton(texto)
            btn.setMinimumSize(QSize(180, 100))
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, g=grados: self._enviar(g))
            grid.addWidget(btn, fila, col)

        return group

    # ── Botón de emergencia ───────────────────────────────────────────────

    def _make_stop_button(self) -> QPushButton:
        btn = QPushButton("■■   PARO DE EMERGENCIA   ■■")
        btn.setObjectName("btn_stop")
        btn.setFixedHeight(90)       # altura fija → siempre visible al pie de pantalla
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Envía 0° — detiene el motor inmediatamente")
        btn.clicked.connect(lambda: self._enviar(0.0))
        return btn

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFrameShadow(QFrame.Shadow.Sunken)
        return f

    def _enviar(self, grados: float) -> None:
        """Publica en /comando_grados y actualiza el display LCD."""
        self._nodo.publicar_grados(grados)

        if grados == 0.0:
            self._lbl_display.setText("ÚLTIMO COMANDO:  *** PARO ***")
            self._lbl_display.setStyleSheet(
                f"background-color:{LCD_BG}; color:#FF4444;"
                "font-family:'Courier New',monospace; font-size:22px; font-weight:bold;"
                f"border-top:3px solid {WIN95_SHADOW}; border-left:3px solid {WIN95_SHADOW};"
                f"border-bottom:3px solid {WIN95_WHITE}; border-right:3px solid {WIN95_WHITE};"
                "padding:10px 20px; letter-spacing:2px;"
            )
        else:
            signo  = "+" if grados > 0 else ""
            color  = "#00FF41" if grados > 0 else "#FFD700"
            self._lbl_display.setText(f"ÚLTIMO COMANDO:  {signo}{grados:.1f}°")
            self._lbl_display.setStyleSheet(
                f"background-color:{LCD_BG}; color:{color};"
                "font-family:'Courier New',monospace; font-size:22px; font-weight:bold;"
                f"border-top:3px solid {WIN95_SHADOW}; border-left:3px solid {WIN95_SHADOW};"
                f"border-bottom:3px solid {WIN95_WHITE}; border-right:3px solid {WIN95_WHITE};"
                "padding:10px 20px; letter-spacing:2px;"
            )

    # ── Cámara Logitech C270 (OpenCV) ───────────────────────────────────

    def _init_camara(self) -> None:
        """
        Abre /dev/video0 (Logitech C270) y arranca el QTimer a ~30 ms (33 FPS).
        Si la cámara no está disponible, el label mostrará SIN SEÑAL y el
        timer seguirá intentando en cada tick.
        """
        self.captura = cv2.VideoCapture(0)   # índice 0 = /dev/video0

        if not self.captura.isOpened():
            self._lbl_camara.setText("🔴 SIN SEÑAL DE CÁMARA")

        # QTimer en hilo Qt — seguro para actualizar widgets
        self._timer_camara = QTimer(self)
        self._timer_camara.timeout.connect(self._actualizar_frame)
        self._timer_camara.start(30)   # 30 ms ≈ 33 FPS

    def _actualizar_frame(self) -> None:
        """
        Llamado cada 30 ms por el QTimer.
        Lee un frame de la cámara, lo convierte a QPixmap y lo muestra en lbl_camara.
        Si la captura falla, muestra el aviso de sin señal.
        """
        if not self.captura.isOpened():
            self._lbl_camara.clear()
            self._lbl_camara.setText("🔴 SIN SEÑAL DE CÁMARA")
            return

        ret, frame = self.captura.read()

        if not ret or frame is None:
            self._lbl_camara.clear()
            self._lbl_camara.setText("🔴 SIN SEÑAL DE CÁMARA")
            return

        # BGR (OpenCV) → RGB (Qt)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        bytes_ln  = ch * w

        # ndarray → QImage → QPixmap
        img    = QImage(frame_rgb.data, w, h, bytes_ln, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        # Escala manteniendo relación de aspecto al tamaño actual del label
        self._lbl_camara.setPixmap(
            pixmap.scaled(
                self._lbl_camara.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ── Limpieza ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """
        Llamado por Qt al cerrar la ventana (ej. desde QApplication.quit()).
        Detiene el timer y libera el puerto USB de la cámara.
        """
        if hasattr(self, '_timer_camara') and self._timer_camara.isActive():
            self._timer_camara.stop()
        if hasattr(self, 'captura') and self.captura.isOpened():
            self.captura.release()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        """Modo kiosco: no se cierra con ninguna tecla."""
        # Escape deshabilitado intencionalmente en modo kiosco.
        super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoGUI()

    hilo_ros = threading.Thread(
        target=rclpy.spin,
        args=(nodo,),
        daemon=True,
        name="hilo_rclpy",
    )
    hilo_ros.start()

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    ventana = VentanaPrincipal(nodo)
    # showFullScreen() se llama dentro de VentanaPrincipal.__init__

    exit_code = app.exec()

    nodo.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
