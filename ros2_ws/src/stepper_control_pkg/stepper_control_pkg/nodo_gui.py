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

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QGroupBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont


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
        # ── MODO KIOSCO: sin decoración de ventana del sistema operativo ──
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._build_ui()
        self.showFullScreen()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Barra de título (sin botón SALIR — modo kiosco)
        main_layout.addWidget(self._make_titlebar())

        # 2. Placeholder de cámara
        main_layout.addWidget(self._make_camera_placeholder())

        # 3. Contenido interior con márgenes
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 12, 10)
        content_layout.setSpacing(10)

        # 3a. Display LCD ancho completo
        content_layout.addWidget(self._make_display())

        # 3b. GroupBox con botones de movimiento
        content_layout.addWidget(self._make_control_group(), stretch=2)

        # 3c. Separador
        content_layout.addWidget(self._hline())

        # 3d. Botón PARO DE EMERGENCIA
        content_layout.addWidget(self._make_stop_button(), stretch=1)

        main_layout.addWidget(content)

    # ── Barra de título (modo kiosco — sin botón cerrar) ─────────────────

    def _make_titlebar(self) -> QWidget:
        """
        Barra de título decorativa estilo Win95 (azul marino).
        No tiene botón de cierre — modo kiosco.
        """
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background-color: {NAVY};")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 14, 0)

        icon = QLabel("■")
        icon.setStyleSheet("color: #AAAAFF; font-size: 14px; background: transparent;")

        title = QLabel("PANEL DE CONTROL  —  ACTUADOR NEMA 17  |  TB6600  |  RPi5")
        title.setObjectName("lbl_titlebar")
        title.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(icon)
        layout.addWidget(title, stretch=1)
        return bar

    # ── Placeholder cámara ──────────────────────────────────────────────

    def _make_camera_placeholder(self) -> QLabel:
        """
        Marco negro con texto verde oscuro que simula un feed de cámara apagado.
        Altura fija 200 px para la orientación vertical 480×800.

        Para conectar el video real (ej. con OpenCV + QImage):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self._lbl_camara.setPixmap(QPixmap.fromImage(img))
        """
        lbl = QLabel()
        lbl.setObjectName("lbl_camara")
        lbl.setFixedHeight(200)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setText(
            "██  CÁMARA DE DETECCIÓN  ██\n\n"
            "[ SIN SEÑAL ]\n\n"
            "Ch.01  480×800  LIVE"
        )
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
        """Grid 2×2 de botones táctiles optimizado para pantalla vertical 480px."""
        group = QGroupBox(" CONTROL DE MOVIMIENTO ")
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
            # En 480 px de ancho con 2 columnas y márgenes: ~220 px c/u
            btn.setMinimumSize(QSize(180, 100))
            btn.setSizePolicy(
                btn.sizePolicy().horizontalPolicy(),
                btn.sizePolicy().verticalPolicy(),
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, g=grados: self._enviar(g))
            grid.addWidget(btn, fila, col)

        return group

    # ── Botón de emergencia ───────────────────────────────────────────────

    def _make_stop_button(self) -> QPushButton:
        btn = QPushButton("■■   PARO DE EMERGENCIA   ■■")
        btn.setObjectName("btn_stop")
        btn.setMinimumHeight(90)
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
