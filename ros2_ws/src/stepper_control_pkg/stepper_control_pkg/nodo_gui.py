#!/usr/bin/env python3
"""
nodo_gui.py
===========
Interfaz gráfica táctil (PyQt6) para controlar el motor NEMA 17 desde una
pantalla de 7 pulgadas (800×480) conectada a la Raspberry Pi 5.

Arquitectura de hilos:
    Hilo principal → Qt event loop (QApplication.exec)
    Hilo daemon    → rclpy.spin(NodoGUI)   [background, nunca bloquea la GUI]

El NodoGUI publica en /comando_grados (Float32). Los botones de la ventana
llaman a nodo.publicar_grados(valor) directamente desde el hilo Qt; publish()
de rclpy es thread-safe, por lo que no se necesitan señales intermedias.

Dependencias:
    pip3 install PyQt6
    sudo apt install python3-rclpy ros-jazzy-std-msgs
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QFrame,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon


# ══════════════════════════════════════════════════════════════════════════════
# Nodo ROS 2 — publicador de /comando_grados
# ══════════════════════════════════════════════════════════════════════════════

class NodoGUI(Node):
    """
    Nodo ROS 2 mínimo que sólo publica en /comando_grados.
    Se ejecuta en un hilo daemon separado del hilo Qt.
    """

    TOPIC = "/comando_grados"

    def __init__(self) -> None:
        super().__init__("nodo_gui")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(Float32, self.TOPIC, qos)
        self.get_logger().info(f"NodoGUI listo — publicando en '{self.TOPIC}'")

    def publicar_grados(self, grados: float) -> None:
        """Publica un valor en /comando_grados. Thread-safe."""
        msg = Float32()
        msg.data = float(grados)
        self._pub.publish(msg)
        self.get_logger().info(f"GUI → {grados:+.1f}°")


# ══════════════════════════════════════════════════════════════════════════════
# Estilos — Dark Mode profesional
# ══════════════════════════════════════════════════════════════════════════════

STYLESHEET = """
/* ── Base ─────────────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'Inter', 'Arial', sans-serif;
}

/* ── Separadores ───────────────────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #21262d;
}

/* ── Labels generales ──────────────────────────────────────────────────── */
QLabel {
    color: #e6edf3;
    background: transparent;
}

/* ── Botones de movimiento ─────────────────────────────────────────────── */
QPushButton {
    background-color: #161b22;
    color: #e6edf3;
    border: 2px solid #30363d;
    border-radius: 14px;
    font-size: 20px;
    font-weight: bold;
    padding: 18px 12px;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: #1c2128;
    border-color: #388bfd;
    color: #79c0ff;
}
QPushButton:pressed {
    background-color: #1f6feb;
    border-color: #58a6ff;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

/* ── Botón PARAR ───────────────────────────────────────────────────────── */
QPushButton#btn_stop {
    background-color: #3d0000;
    color: #ff6b6b;
    border: 2px solid #da3633;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 2px;
    border-radius: 14px;
    padding: 22px 12px;
}
QPushButton#btn_stop:hover {
    background-color: #da3633;
    color: #ffffff;
    border-color: #f85149;
}
QPushButton#btn_stop:pressed {
    background-color: #b91c1c;
    border-color: #ff6b6b;
    color: #ffffff;
}

/* ── Label de título ───────────────────────────────────────────────────── */
QLabel#lbl_titulo {
    font-size: 22px;
    font-weight: bold;
    color: #79c0ff;
    letter-spacing: 1px;
}

/* ── Label de subtítulo / estado ───────────────────────────────────────── */
QLabel#lbl_estado {
    font-size: 14px;
    color: #8b949e;
    letter-spacing: 0.5px;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Ventana principal
# ══════════════════════════════════════════════════════════════════════════════

class VentanaPrincipal(QMainWindow):
    """
    Ventana táctil optimizada para 800×480 px (pantalla de 7 pulgadas).

    Layout:
        ┌──────────────────────────────────────┐
        │  [Logo Pinguinin]   Control Motor    │  ← Barra superior
        ├──────────────────────────────────────┤
        │  [ +90° ]   [ -90° ]                │  ← Fila 1 de control
        │  [+360°]   [-360°]                  │  ← Fila 2 de control
        ├──────────────────────────────────────┤
        │         ●  PARAR  ●                  │  ← Botón de emergencia
        ├──────────────────────────────────────┤
        │  Último comando: ---                 │  ← Barra de estado
        └──────────────────────────────────────┘
    """

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self._setup_window()
        self._setup_ui()

    # ── Configuración de ventana ──────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("Control Motor NEMA 17 — Pinguinin")
        self.setMinimumSize(800, 480)
        self.resize(800, 480)
        # En la RPi5 con pantalla táctil, descomenta la siguiente línea:
        # self.showFullScreen()

    # ── Construcción de UI ────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        layout_raiz = QVBoxLayout(widget_central)
        layout_raiz.setContentsMargins(16, 12, 16, 12)
        layout_raiz.setSpacing(12)

        # ── 1. Barra superior (logo + título) ─────────────────────────────
        layout_raiz.addLayout(self._crear_barra_superior())

        # ── Separador ─────────────────────────────────────────────────────
        layout_raiz.addWidget(self._separador())

        # ── 2. Grid de botones de control ─────────────────────────────────
        layout_raiz.addLayout(self._crear_grid_botones(), stretch=3)

        # ── Separador ─────────────────────────────────────────────────────
        layout_raiz.addWidget(self._separador())

        # ── 3. Botón PARAR ────────────────────────────────────────────────
        layout_raiz.addWidget(self._crear_boton_stop(), stretch=1)

        # ── 4. Barra de estado ────────────────────────────────────────────
        layout_raiz.addWidget(self._separador())
        layout_raiz.addWidget(self._crear_barra_estado())

    def _crear_barra_superior(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(16)

        # ── Espacio para el logo de Pinguinin ──────────────────────────────
        # Para activarlo: copia el archivo del logo en el paquete y descomenta.
        self._lbl_logo = QLabel()
        self._lbl_logo.setFixedSize(60, 60)
        self._lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_logo.setStyleSheet(
            "border: 1px solid #30363d; border-radius: 8px; background: #161b22;"
        )
        # ── LOGO: descomenta y ajusta la ruta cuando tengas el archivo ────
        # pixmap = QPixmap("/path/to/pinguinin_logo.png")
        # self._lbl_logo.setPixmap(
        #     pixmap.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio,
        #                   Qt.TransformationMode.SmoothTransformation)
        # )
        # ─────────────────────────────────────────────────────────────────

        # Placeholder visual mientras no hay logo
        self._lbl_logo.setText("🐧")
        self._lbl_logo.setFont(QFont("Segoe UI Emoji", 26))
        layout.addWidget(self._lbl_logo)

        # ── Título y subtítulo ─────────────────────────────────────────────
        layout_texto = QVBoxLayout()
        layout_texto.setSpacing(2)

        lbl_titulo = QLabel("Control Motor NEMA 17")
        lbl_titulo.setObjectName("lbl_titulo")
        layout_texto.addWidget(lbl_titulo)

        lbl_sub = QLabel("Driver TB6600 · Raspberry Pi 5 · ROS 2 Jazzy")
        lbl_sub.setObjectName("lbl_estado")
        layout_texto.addWidget(lbl_sub)

        layout.addLayout(layout_texto)
        layout.addStretch()
        return layout

    def _crear_grid_botones(self) -> QGridLayout:
        """
        Grid 2×2 con los botones de movimiento relativo.
        Tamaño mínimo: 170×90 px para ser cómodos con el dedo en 7".
        """
        grid = QGridLayout()
        grid.setSpacing(12)

        definiciones = [
            # (texto, grados, fila, columna, tooltip)
            ("+90°",        90.0,   0, 0, "Girar 90° en sentido horario"),
            ("-90°",        -90.0,  0, 1, "Girar 90° en sentido antihorario"),
            ("+1 Vuelta",   360.0,  1, 0, "Girar 1 vuelta completa (360°) horario"),
            ("-1 Vuelta",   -360.0, 1, 1, "Girar 1 vuelta completa (360°) antihorario"),
        ]

        for texto, grados, fila, col, tooltip in definiciones:
            btn = QPushButton(texto)
            btn.setToolTip(tooltip)
            btn.setMinimumSize(QSize(170, 90))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # Captura 'grados' por valor en el lambda con argumento por defecto
            btn.clicked.connect(lambda _, g=grados: self._enviar_grados(g))
            grid.addWidget(btn, fila, col)

        return grid

    def _crear_boton_stop(self) -> QPushButton:
        btn_stop = QPushButton("⬛  PARAR")
        btn_stop.setObjectName("btn_stop")
        btn_stop.setMinimumHeight(80)
        btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_stop.setToolTip("Detener el motor inmediatamente (envía 0°)")
        btn_stop.clicked.connect(lambda: self._enviar_grados(0.0))
        return btn_stop

    def _crear_barra_estado(self) -> QLabel:
        self._lbl_estado = QLabel("Último comando:  —")
        self._lbl_estado.setObjectName("lbl_estado")
        self._lbl_estado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return self._lbl_estado

    @staticmethod
    def _separador() -> QFrame:
        linea = QFrame()
        linea.setFrameShape(QFrame.Shape.HLine)
        linea.setFrameShadow(QFrame.Shadow.Sunken)
        return linea

    # ── Lógica de publicación ─────────────────────────────────────────────

    def _enviar_grados(self, grados: float) -> None:
        """
        Publica el valor en /comando_grados y actualiza la barra de estado.
        Corre en el hilo Qt (main thread); publish() de rclpy es thread-safe.
        """
        self._nodo.publicar_grados(grados)

        if grados == 0.0:
            texto_estado = "Último comando:  PARAR (0°)"
            color = "#f85149"
        else:
            signo = "+" if grados > 0 else ""
            texto_estado = f"Último comando:  {signo}{grados:.1f}°"
            color = "#56d364" if grados > 0 else "#ffa657"

        self._lbl_estado.setText(texto_estado)
        self._lbl_estado.setStyleSheet(f"color: {color}; font-size: 14px;")


# ══════════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    """
    Orden de arranque:
        1. rclpy.init()
        2. NodoGUI() — crea el nodo y el publicador
        3. hilo daemon → rclpy.spin(nodo)   [background]
        4. QApplication + VentanaPrincipal  [main thread]
        5. Al cerrar la ventana → destroy_node() + rclpy.shutdown()
    """
    rclpy.init(args=args)
    nodo = NodoGUI()

    # ── Hilo daemon de ROS 2 ──────────────────────────────────────────────
    hilo_ros = threading.Thread(
        target=rclpy.spin,
        args=(nodo,),
        daemon=True,        # muere automáticamente cuando el proceso Qt termina
        name="hilo_rclpy",
    )
    hilo_ros.start()

    # ── Aplicación Qt en el hilo principal ───────────────────────────────
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    ventana = VentanaPrincipal(nodo)
    ventana.show()

    exit_code = app.exec()   # bloquea hasta que se cierra la ventana

    # ── Limpieza ──────────────────────────────────────────────────────────
    nodo.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
