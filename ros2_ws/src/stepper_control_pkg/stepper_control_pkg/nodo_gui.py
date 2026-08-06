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

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
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
COLOR_ERROR  = "#B91C1C"

ESTILO_FRAME_NEUTRO = f"""
QFrame#frame_principal {{
    background-color: {WIN95_GRAY};
}}
"""

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
        self.bloqueo_error: bool = False
        self._error_peso_persistente: bool = False

        self._timer_error = QTimer(self)
        self._timer_error.setSingleShot(True)
        self._timer_error.setInterval(4000)
        self._timer_error.timeout.connect(self._reset_ui)

        self._timer_exito = QTimer(self)
        self._timer_exito.setSingleShot(True)
        self._timer_exito.setInterval(2000)
        self._timer_exito.timeout.connect(self._reset_ui)

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
        self.frame_principal = QFrame()
        self.frame_principal.setObjectName("frame_principal")
        self.frame_principal.setStyleSheet(ESTILO_FRAME_NEUTRO)
        self.setCentralWidget(self.frame_principal)
        layout = QVBoxLayout(self.frame_principal)
        
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

        # Alias semanticos: estos son los dos textos de estado que deben
        # actualizarse siempre como una sola transaccion visual.
        self.lbl_estado = self.lbl_banner
        self.lbl_instruccion = self.lbl_veredicto_inferior

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
        if self.bloqueo_error:
            return
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

    def _reset_ui(self) -> None:
        """
        Restaura la interfaz visual al estado neutro tras un rechazo, error o exito.
        """
        self._timer_error.stop()
        self._timer_exito.stop()
        self._ultimo_estado = ""
        banner_bg = "#808080"
        inferior_color = "#808080"

        # Eliminar por completo el CSS del estado anterior antes de aplicar el
        # estilo neutro. Esto evita que reglas rojas/verdes queden acumuladas.
        self.frame_principal.setStyleSheet("")
        self.frame_principal.setStyleSheet(ESTILO_FRAME_NEUTRO)

        self.lbl_banner.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: {banner_bg}; padding: 10px;"
        )
        self.lbl_veredicto_inferior.setStyleSheet(
            f"font-size: 20pt; font-weight: bold; color: {inferior_color};"
        )
        self.lbl_banner.setText("Esperando envase... Coloque el objeto en el sistema.")
        self.lbl_veredicto_inferior.setText("Inserte una lata o botella en el centro.")
        self.btn_confirmar.setVisible(False)
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        self.bloqueo_error = False
        self._error_peso_persistente = False

    @staticmethod
    def _es_rechazo_peso(estado: str) -> bool:
        texto = estado.casefold().replace("-", "_")
        return (
            "rechazo_por_peso" in texto
            or "rechazado_exceso_peso" in texto
            or "exceso de peso" in texto
            or "exceso_peso" in texto
        )

    @staticmethod
    def _es_estado_rearme(estado: str) -> bool:
        texto = estado.casefold()
        return (
            "sistema disponible" in texto
            or "esperando envase" in texto
            or "esperando elemento" in texto
        )

    @staticmethod
    def _es_estado_error(estado: str) -> bool:
        """Reconoce rechazos estructurados y errores directos de cualquier nodo."""
        texto = estado.casefold().replace("-", "_")
        marcadores_error = (
            "rechaz",
            "error",
            "exceso_peso",
            "exceso de peso",
            "fallo",
            "timeout",
        )
        return any(marcador in texto for marcador in marcadores_error)

    def _mostrar_error(self, estado: str) -> None:
        """Muestra un estado rojo atomico y bloquea mensajes durante 4 segundos."""
        texto = estado.casefold()
        self._timer_exito.stop()
        self.bloqueo_error = True
        self._error_peso_persistente = self._es_rechazo_peso(estado)
        self._ultimo_estado = estado

        # Mantener neutro el frame general: sólo el banner y la instrucción
        # representan visualmente el rechazo.
        self.frame_principal.setStyleSheet("")
        self.frame_principal.setStyleSheet(ESTILO_FRAME_NEUTRO)
        estilo_estado_error = (
            f"font-size: 24px; font-weight: bold; color: #FFFFFF; "
            f"background-color: {COLOR_ERROR}; padding: 10px;"
        )
        estilo_instruccion_error = (
            "font-size: 20pt; font-weight: bold; color: #FFFFFF; "
            f"background-color: {COLOR_ERROR}; padding: 8px;"
        )
        self.lbl_estado.setStyleSheet(estilo_estado_error)
        self.lbl_instruccion.setStyleSheet(estilo_instruccion_error)

        if "balanza" in texto:
            self.lbl_estado.setText("ERROR DE BALANZA")
            self.lbl_instruccion.setText(
                "No fue posible validar el peso. Retire el envase y avise a un encargado."
            )
        elif "actuador" in texto or "mecanismo" in texto or "timeout" in texto:
            self.lbl_estado.setText("ERROR DE MECANISMO")
            self.lbl_instruccion.setText(
                "El mecanismo no pudo completar el ciclo. Retire el envase y espere."
            )
        else:
            self.lbl_estado.setText("ENVASE RECHAZADO")
            self.lbl_instruccion.setText(
                "Exceso de peso o suciedad detectada.\nPor favor, retire el envase."
            )

        self.btn_confirmar.setVisible(False)
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        if self._error_peso_persistente:
            self._timer_error.stop()
        else:
            self._timer_error.start()

    def _actualizar_estado(self, estado: str) -> None:
        # Durante la pantalla roja se descartan incluso mensajes validos de la
        # camara. Asi ningun estado tardio puede repintar la interfaz en verde.
        if self.bloqueo_error:
            if self._error_peso_persistente and self._es_estado_rearme(estado):
                self._reset_ui()
            return

        if self._ultimo_estado == estado:
            return

        es_exito = "reciclaje_exitoso" in estado.casefold()
        if self._timer_exito.isActive() and not es_exito:
            self._timer_exito.stop()

        if self._es_estado_error(estado):
            self._mostrar_error(estado)
            return

        self._ultimo_estado = estado
        
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.setStyleSheet(self._estilo_btn_inactivo)
        self.btn_confirmar.setVisible(False)
        
        texto = estado.lower()

        # Parseo de mensajes estructurados o directos de backend
        if "reciclaje_exitoso" in texto:
            banner_txt = "RECICLAJE COMPLETADO"
            veredicto_txt = "Envase procesado correctamente."
            boton_txt = "NONE"
        elif estado.count("|") >= 2:
            partes = estado.split("|", 2)
            banner_txt = partes[0]
            veredicto_txt = partes[1]
            boton_txt = partes[2]
        elif "|" in estado:
            banner_txt, veredicto_txt = estado.split("|", 1)
            boton_txt = "NONE"
        else:
            banner_txt = "ESTADO DEL SISTEMA"
            veredicto_txt = estado
            boton_txt = "NONE"

        # Determinacion sistematica de colores
        banner_upper = banner_txt.upper()
        veredicto_upper = veredicto_txt.upper()

        if "RECHAZ" in banner_upper or "RECHAZ" in veredicto_upper or "ERROR" in banner_upper:
            banner_bg = "#B91C1C"       # Rojo de rechazo / error
            inferior_color = "#B91C1C"
        elif "ACEPTADA" in veredicto_upper or "ACEPTADO" in veredicto_upper or "EXITOSO" in banner_upper:
            banner_bg = "#15803D"       # Verde de aceptacion
            inferior_color = "#15803D"
        elif "DISPONIBLE" in banner_upper or "ESPERANDO" in banner_upper:
            banner_bg = "#808080"       # Gris de reposo
            inferior_color = "#808080"
        elif "MOVIMIENTO" in banner_upper or "PROCESANDO" in banner_upper or "ESTABILIZANDO" in veredicto_upper:
            banner_bg = "#2563EB"       # Azul de operacion en curso
            inferior_color = "#2563EB"
        else:
            banner_bg = "#B45309"       # Ambar de advertencia
            inferior_color = "#B45309"

        self.lbl_banner.setStyleSheet(f"font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: {banner_bg}; padding: 10px;")
        self.lbl_veredicto_inferior.setStyleSheet(f"font-size: 20pt; font-weight: bold; color: {inferior_color};")
        self.lbl_banner.setText(banner_txt)
        self.lbl_veredicto_inferior.setText(veredicto_txt)

        if boton_txt != "NONE":
            self.btn_confirmar.setText(boton_txt)
            self.btn_confirmar.setVisible(True)
            self.btn_confirmar.setEnabled(True)
            self.btn_confirmar.setStyleSheet(self._estilo_btn_activo)
        else:
            self.btn_confirmar.setVisible(False)
            self.btn_confirmar.setEnabled(False)

        if es_exito:
            # Actuadores publica el exito despues de que vision ya regreso a
            # BUSQUEDA. Sin este timer, ese mensaje tardio queda fijo para siempre.
            self._timer_exito.start()

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
