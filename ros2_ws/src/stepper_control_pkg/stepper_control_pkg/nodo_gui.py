#!/usr/bin/env python3
"""
nodo_gui.py  —  Dashboard Informativo Pasivo (PyQt6)
====================================================
Modo Producción — pantalla vertical 480x800, sin bordes (FramelessWindowHint).
Estética dashboard, split screen para video raw y video segmentado.
"""

import sys
import threading
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSizePolicy, QFrame
)

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
QLabel {{
    color: #000000;
}}
QLabel#lbl_banner {{
    border-top:    3px solid {WIN95_SHADOW};
    border-left:   3px solid {WIN95_SHADOW};
    border-bottom: 3px solid {WIN95_WHITE};
    border-right:  3px solid {WIN95_WHITE};
}}
QLabel#lbl_camara_raw, QLabel#lbl_camara_seg {{
    background-color: #000000;
    border-top:    4px solid {WIN95_DARK};
    border-left:   4px solid {WIN95_DARK};
    border-bottom: 4px solid {WIN95_WHITE};
    border-right:  4px solid {WIN95_WHITE};
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {WIN95_SHADOW};
}}
"""

QOS_VIDEO = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)


class NodoGUI(Node):
    def __init__(self) -> None:
        super().__init__("nodo_gui")
        self._bridge = CvBridge()

        # Thread safe callbacks
        self._fn_lock = threading.Lock()
        self._fn_frame_raw = None
        self._fn_frame_seg = None
        self._fn_peso = None
        self._fn_area = None
        self._fn_estado = None

        self.create_subscription(Image, "/camara/video_raw", self._cb_camara_raw, QOS_VIDEO)
        self.create_subscription(Image, "/camara/video_segmentado", self._cb_camara_seg, QOS_VIDEO)
        self.create_subscription(Float32, "/peso_botella", self._cb_peso, 10)
        self.create_subscription(Float32, "/tamano_estimado", self._cb_area, 10)
        self.create_subscription(String, "/analisis_botella", self._cb_estado, 10)

        self.get_logger().info("NodoGUI (Dashboard) iniciado de forma pasiva.")

    def registrar_callbacks(self, fn_raw, fn_seg, fn_peso, fn_area, fn_estado):
        with self._fn_lock:
            self._fn_frame_raw = fn_raw
            self._fn_frame_seg = fn_seg
            self._fn_peso = fn_peso
            self._fn_area = fn_area
            self._fn_estado = fn_estado

    def _cb_camara_raw(self, msg: Image) -> None:
        with self._fn_lock:
            fn = self._fn_frame_raw
        if fn:
            img = self._msg_to_qimage(msg)
            if img: fn(img)

    def _cb_camara_seg(self, msg: Image) -> None:
        with self._fn_lock:
            fn = self._fn_frame_seg
        if fn:
            img = self._msg_to_qimage(msg)
            if img: fn(img)

    def _cb_peso(self, msg: Float32) -> None:
        with self._fn_lock:
            fn = self._fn_peso
        if fn: fn(msg.data)

    def _cb_area(self, msg: Float32) -> None:
        with self._fn_lock:
            fn = self._fn_area
        if fn: fn(msg.data)

    def _cb_estado(self, msg: String) -> None:
        with self._fn_lock:
            fn = self._fn_estado
        if fn: fn(msg.data)

    def _msg_to_qimage(self, msg: Image) -> QImage:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            h, w, ch = frame.shape
            return QImage(frame.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        except Exception as exc:
            self.get_logger().warn(f"Error procesando frame: {exc}", once=True)
            return None


class VentanaPrincipal(QMainWindow):
    senal_frame_raw = pyqtSignal(QImage)
    senal_frame_seg = pyqtSignal(QImage)
    senal_peso = pyqtSignal(float)
    senal_area = pyqtSignal(float)
    senal_estado = pyqtSignal(str)

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self.setWindowTitle("Dashboard Informativo")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)

        self._build_ui()
        self._ultimo_peso = 0.0
        self._ultimo_estado = "vacio"
        self._conteo_histeresis = 0
        self._conteo_bloqueo = 0
        self._estado_visual_actual = 'ESPERA'

        self.senal_frame_raw.connect(self._actualizar_raw)
        self.senal_frame_seg.connect(self._actualizar_seg)
        self.senal_peso.connect(self._actualizar_peso)
        self.senal_area.connect(self._actualizar_area)
        self.senal_estado.connect(self._actualizar_estado)

        nodo.registrar_callbacks(
            self.senal_frame_raw.emit,
            self.senal_frame_seg.emit,
            self.senal_peso.emit,
            self.senal_area.emit,
            self.senal_estado.emit
        )

        self.showFullScreen()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        
        # Banner UX Superior
        self.lbl_banner = QLabel("Esperando botella... Ingrese su envase.")
        self.lbl_banner.setObjectName("lbl_banner")
        self.lbl_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #808080; padding: 10px;")
        layout.addWidget(self.lbl_banner)

        # Videos en Grid
        grid_videos = QGridLayout()
        self.lbl_camara_raw = QLabel("[ VIDEO RAW ]")
        self.lbl_camara_raw.setObjectName("lbl_camara_raw")
        self.lbl_camara_seg = QLabel("[ VIDEO SEGMENTADO ]")
        self.lbl_camara_seg.setObjectName("lbl_camara_seg")
        
        for lbl in (self.lbl_camara_raw, self.lbl_camara_seg):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lbl.setMinimumSize(1, 1)
            
        grid_videos.addWidget(self.lbl_camara_raw, 0, 0)
        grid_videos.addWidget(self.lbl_camara_seg, 0, 1)
        layout.addLayout(grid_videos, stretch=3)

        # Panel de Metadatos Inferior
        panel_inferior = QHBoxLayout()
        
        self.lbl_peso = self._make_data_label("PESO (g)", "0.0")
        self.lbl_area = self._make_data_label("ÁREA (cm²)", "0.0")
        self.lbl_estado = self._make_data_label("ESTADO", "Vacio")
        
        panel_inferior.addWidget(self.lbl_peso)
        panel_inferior.addWidget(self.lbl_area)
        panel_inferior.addWidget(self.lbl_estado)
        
        layout.addLayout(panel_inferior, stretch=1)

    def _make_data_label(self, titulo: str, valor: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        
        lbl_tit = QLabel(titulo)
        lbl_tit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_tit.setStyleSheet("font-size: 14px; font-weight: bold; color: #000000;")
        
        lbl_val = QLabel(valor)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_val.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {LCD_FG}; background-color: {LCD_BG}; padding: 5px; border: 2px inset {WIN95_SHADOW};")
        
        layout.addWidget(lbl_tit)
        layout.addWidget(lbl_val)
        
        # Guardamos la referencia dinámica en el contenedor para poder modificarla después
        container.valor_label = lbl_val
        container.setStyleSheet(f"background-color: {WIN95_GRAY}; border: 2px outset {WIN95_WHITE}; margin: 2px;")
        return container

    def _actualizar_raw(self, img: QImage) -> None:
        if self.lbl_camara_raw.size().width() > 0 and self.lbl_camara_raw.size().height() > 0:
            self.lbl_camara_raw.setPixmap(QPixmap.fromImage(img).scaled(
                self.lbl_camara_raw.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))

    def _actualizar_seg(self, img: QImage) -> None:
        if self.lbl_camara_seg.size().width() > 0 and self.lbl_camara_seg.size().height() > 0:
            self.lbl_camara_seg.setPixmap(QPixmap.fromImage(img).scaled(
                self.lbl_camara_seg.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))

    def _actualizar_peso(self, peso: float) -> None:
        self._ultimo_peso = peso
        self.lbl_peso.valor_label.setText(f"{peso:.1f}")
        self._evaluar_banner()

    def _actualizar_area(self, area: float) -> None:
        self.lbl_area.valor_label.setText(f"{area:.1f}")

    def _actualizar_estado(self, estado: str) -> None:
        self._ultimo_estado = estado
        self.lbl_estado.valor_label.setText(estado.upper())
        self._evaluar_banner()

    def _evaluar_banner(self) -> None:
        # Si estamos en un estado final, reducimos el contador de bloqueo y no hacemos nada hasta que termine
        if self._conteo_bloqueo > 0:
            self._conteo_bloqueo -= 1
            return

        peso = self._ultimo_peso
        estado = self._ultimo_estado.lower()

        # Determinamos el estado físico instantáneo
        if peso < 10.0:
            estado_inst = 'ESPERA'
        else:
            if estado in ["grande", "chica"]:
                if (estado == "chica" and peso < 40.0) or (estado == "grande" and peso < 80.0):
                    estado_inst = 'LIMPIA'
                else:
                    estado_inst = 'SUCIA'
            else:
                estado_inst = 'ANALIZANDO'

        # Lógica de histéresis (Debouncing)
        if estado_inst == self._estado_visual_actual:
            self._conteo_histeresis = 0
        else:
            self._conteo_histeresis += 1

            # Umbral de confirmación para cambiar de estado visual (15 cuadros)
            if self._conteo_histeresis >= 15:
                self._estado_visual_actual = estado_inst
                self._conteo_histeresis = 0

                # Aplicar el nuevo estado a la UI
                if estado_inst == 'ESPERA':
                    self.lbl_banner.setText("Esperando botella... Ingrese su envase.")
                    self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #808080; padding: 10px;")
                elif estado_inst == 'ANALIZANDO':
                    self.lbl_banner.setText("Analizando botella... Por favor espere.")
                    self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #808000; padding: 10px;")
                elif estado_inst == 'LIMPIA':
                    self.lbl_banner.setText("¡La botella está perfecta! Proceso realizado.")
                    self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #008000; padding: 10px;")
                    self._conteo_bloqueo = 90  # Congelar ~3 segundos
                elif estado_inst == 'SUCIA':
                    self.lbl_banner.setText("La botella presenta suciedad. Favor retirarla de la máquina.")
                    self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #800000; padding: 10px;")
                    self._conteo_bloqueo = 90  # Congelar ~3 segundos

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoGUI()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    ventana = VentanaPrincipal(nodo)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(nodo)
    hilo_ros = threading.Thread(target=executor.spin, daemon=True)
    hilo_ros.start()

    exit_code = app.exec()
    executor.shutdown(timeout_sec=2.0)
    hilo_ros.join(timeout=3.0)
    nodo.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
