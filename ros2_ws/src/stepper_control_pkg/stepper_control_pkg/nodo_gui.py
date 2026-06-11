#!/usr/bin/env python3
"""
nodo_gui.py — Interfaz HMI pasiva de monitoreo en tiempo real (PyQt6).

Implementa un dashboard de visualización para sistemas de clasificación
autónoma. Opera en pantalla vertical 480×800 sin bordes (FramelessWindowHint),
mostando en tiempo real los flujos de video RAW y segmentado, el peso del
sensor de carga y el estado de clasificación del sistema.
"""

import sys
import threading
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, String, Bool, Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSizePolicy, QFrame, QPushButton
)

WIN95_GRAY   = "#D4D0C8"
WIN95_WHITE  = "#FFFFFF"
WIN95_LIGHT  = "#EBEBEB"
WIN95_SHADOW = "#808080"
WIN95_DARK   = "#404040"
LCD_BG       = "#001A00"
LCD_FG       = "#00FF41"
ANCHO_VISOR  = 400
ALTO_VISOR   = 300

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
    """Nodo ROS 2 pasivo de interfaz HMI para sistemas de visión autónoma.

    Actua exclusivamente como puente ROS↔Qt: se suscribe a los tópicos de
    datos del sistema y reenvía la información a la capa de presentación
    mediante señales thread-safe. No contiene lógica de negocio.

    Suscripciones:
        /camara/video_procesado (sensor_msgs/Image): Flujo de video anotado.
        /camara/video_segmentado (sensor_msgs/Image): Mapa de anomalías HSV.
        /peso_elemento (std_msgs/Float32): Peso en gramos del sensor de carga.
        /tamano_estimado (std_msgs/Float32): Área estimada del objetivo en cm².
        /clasificacion_objeto (std_msgs/String): Veredicto de clasificación.
    """

    def __init__(self) -> None:
        """Inicializa suscripciones ROS 2 y el mecanismo de callbacks thread-safe."""
        super().__init__("nodo_gui")
        self._bridge = CvBridge()

        # Thread safe callbacks
        self._fn_lock = threading.Lock()
        self._fn_frame_raw = None
        self._fn_frame_seg = None
        self._fn_estado = None

        self.create_subscription(Image, "/camara/video_procesado", self._cb_camara_raw, QOS_VIDEO)
        self.create_subscription(Image, "/camara/video_segmentado", self._cb_camara_seg, QOS_VIDEO)
        self.create_subscription(String, "/clasificacion_objeto", self._cb_estado, 10)

        # Buffer y Timer para Throttle de Video a 15 FPS
        self._pending_raw = None
        self._pending_seg = None
        self.create_timer(1.0 / 15.0, self._flush_frames)

        self.pub_confirmacion = self.create_publisher(Empty, "/ui/confirmacion", 10)

        self.get_logger().info("NodoGUI (Dashboard) iniciado de forma pasiva.")

    def publicar_confirmacion(self):
        self.pub_confirmacion.publish(Empty())

    def registrar_callbacks(self, fn_raw, fn_seg, fn_estado):
        with self._fn_lock:
            self._fn_frame_raw = fn_raw
            self._fn_frame_seg = fn_seg
            self._fn_estado = fn_estado

    def _cb_camara_raw(self, msg: Image) -> None:
        self._pending_raw = msg

    def _cb_camara_seg(self, msg: Image) -> None:
        self._pending_seg = msg

    def _flush_frames(self) -> None:
        with self._fn_lock:
            fn_raw = self._fn_frame_raw
            fn_seg = self._fn_frame_seg
        
        if fn_raw and self._pending_raw:
            img = self._msg_to_qimage(self._pending_raw)
            if img: fn_raw(img)
            self._pending_raw = None
            
        if fn_seg and self._pending_seg:
            img = self._msg_to_qimage(self._pending_seg)
            if img: fn_seg(img)
            self._pending_seg = None



    def _cb_estado(self, msg: String) -> None:
        with self._fn_lock:
            fn = self._fn_estado
        if fn: fn(msg.data)

    def _msg_to_qimage(self, msg: Image) -> QImage:
        try:
            # Conversion directa desde el buffer RGB8, evita CvBridge y Numpy copies redundantes
            return QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format.Format_RGB888).copy()
        except Exception as exc:
            self.get_logger().warn(f"Error procesando frame: {exc}", once=True)
            return None


class VentanaPrincipal(QMainWindow):
    """Ventana principal del dashboard HMI (PyQt6).

    Implementa la capa de presentación del sistema de clasificación.
    Recibe datos desde NodoGUI vía señales Qt (thread-safe) y actualiza
    los widgets en el hilo principal de Qt. Gestiona el ciclo de estados
    visuales: ESPERA → ANALIZANDO → ESTADO_OPTIMO / ANOMALIA_DETECTADA.

    Attributes:
        senal_frame_raw: Señal para actualizar el visor de video RAW.
        senal_frame_seg: Señal para actualizar el visor de segmentación.
        senal_estado: Señal para actualizar el banner de estado.
    """

    senal_frame_raw = pyqtSignal(QImage)
    senal_frame_seg = pyqtSignal(QImage)
    senal_estado = pyqtSignal(str)

    def __init__(self, nodo: NodoGUI) -> None:
        super().__init__()
        self._nodo = nodo
        self.setWindowTitle("Dashboard Informativo")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._build_ui()
        self._ultimo_estado = "vacio"

        self.senal_frame_raw.connect(self._actualizar_raw)
        self.senal_frame_seg.connect(self._actualizar_seg)
        self.senal_estado.connect(self._actualizar_estado)

        nodo.registrar_callbacks(
            self.senal_frame_raw.emit,
            self.senal_frame_seg.emit,
            self.senal_estado.emit
        )

        self.showFullScreen()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        
        # Banner UX Superior
        self.lbl_banner = QLabel("Esperando elemento... Coloque el objeto en el sistema.")
        self.lbl_banner.setObjectName("lbl_banner")
        self.lbl_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_banner.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #808080; padding: 10px;")
        self.lbl_banner.setWordWrap(True)
        layout.addWidget(self.lbl_banner)

        # Videos en Layout Vertical
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camara_raw = QLabel("[ VIDEO RAW ]")
        self.lbl_camara_raw.setObjectName("lbl_camara_raw")
        self.lbl_camara_seg = QLabel("[ VIDEO SEGMENTADO ]")
        self.lbl_camara_seg.setObjectName("lbl_camara_seg")
        
        for lbl in (self.lbl_camara_raw, self.lbl_camara_seg):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(ANCHO_VISOR, ALTO_VISOR)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lbl.setScaledContents(False)
            
        layout.addWidget(self.lbl_camara_raw, alignment=Qt.AlignmentFlag.AlignCenter)
        
        separador = QFrame()
        separador.setFrameShape(QFrame.Shape.HLine)
        separador.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separador)
        
        layout.addWidget(self.lbl_camara_seg, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_veredicto_inferior = QLabel("")
        self.lbl_veredicto_inferior.setObjectName("lbl_veredicto_inferior")
        self.lbl_veredicto_inferior.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold;")
        self.lbl_veredicto_inferior.setWordWrap(True)
        layout.addWidget(self.lbl_veredicto_inferior)

        self.btn_confirmar = QPushButton("CONFIRMAR Y GIRAR MOTOR")
        self.btn_confirmar.setObjectName("btn_confirmar")
        self._estilo_btn_inactivo = "background-color: #D3D3D3; color: #696969; font-size: 18pt; font-weight: bold; border-radius: 10px; padding: 15px;"
        self._estilo_btn_activo = "background-color: #006400; color: #FFFFFF; font-size: 18pt; font-weight: bold; border-radius: 10px; padding: 15px;"
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setVisible(False)
        self.btn_confirmar.clicked.connect(self._on_confirmar_clicked)
        layout.addWidget(self.btn_confirmar)

    def _on_confirmar_clicked(self) -> None:
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        self.btn_confirmar.setVisible(False)
        self._nodo.publicar_confirmacion()

    def _actualizar_raw(self, img: QImage) -> None:
        self.lbl_camara_raw.setPixmap(QPixmap.fromImage(img).scaled(
            ANCHO_VISOR, ALTO_VISOR, Qt.AspectRatioMode.KeepAspectRatio
        ))

    def _actualizar_seg(self, img: QImage) -> None:
        self.lbl_camara_seg.setPixmap(QPixmap.fromImage(img).scaled(
            ANCHO_VISOR, ALTO_VISOR, Qt.AspectRatioMode.KeepAspectRatio
        ))

    def _actualizar_estado(self, estado: str) -> None:
        if self._ultimo_estado == estado:
            return
        self._ultimo_estado = estado
        
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        self.btn_confirmar.setVisible(False)
        
        if "|" in estado:
            banner_txt, veredicto_txt = estado.split("|", 1)
        else:
            banner_txt = "ESTADO: DESCONOCIDO"
            veredicto_txt = estado

        if "DISPONIBLE" in banner_txt:
            self.lbl_banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #808080; padding: 10px;")
            self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold; color: #808080;")
        elif "RETORNANDO" in banner_txt:
            self.lbl_banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #FFA500; padding: 10px;")
            self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold; color: #FFA500;")
        elif "ACEPTADA" in veredicto_txt.upper():
            self.lbl_banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #008000; padding: 10px;")
            self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold; color: #008000;")
            self.btn_confirmar.setVisible(True)
            self.btn_confirmar.setEnabled(True)
            self.btn_confirmar.setStyleSheet(self._estilo_btn_activo)
        elif "RECHAZADA" in veredicto_txt.upper():
            self.lbl_banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #800000; padding: 10px;")
            self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold; color: #800000;")
        else:
            # Procesando envase
            self.lbl_banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #808000; padding: 10px;")
            self.lbl_veredicto_inferior.setStyleSheet("font-size: 20pt; font-weight: bold; color: #808000;")

        self.lbl_banner.setText(banner_txt)
        self.lbl_veredicto_inferior.setText(veredicto_txt)

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
