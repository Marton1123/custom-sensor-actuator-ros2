#!/usr/bin/env python3
"""
nodo_camara.py — Nodo ROS 2 de captura de video (Logitech C270)
Publica sensor_msgs/Image en /camara/video_raw a ~30 FPS usando CvBridge.
Dependencias: sudo apt install python3-opencv ros-jazzy-cv-bridge

Resiliencia:
    Si la camara falla despues de abrirse (ej. cable USB desconectado), el nodo
    intenta reconectarse automaticamente tras MAX_FALLOS_CONSECUTIVOS frames
    fallidos. La GUI mostrara el placeholder de texto durante la reconexion.
"""

import cv2
import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

QOS_VIDEO = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

DEVICE_INDEX         = 0
TIMER_PERIOD         = 1.0 / 30   # ~33 ms → 30 FPS
MAX_FALLOS_CONSECUTIVOS = 30      # ~1 segundo de fallos antes de reconectar


class NodoCamara(Node):
    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._pub_analisis = self.create_publisher(String, "/analisis_botella", 10)

        # Parametros de IA (YOLO)
        self.declare_parameter("k_area", 0.05)
        self.declare_parameter("conf_threshold", 0.70)
        
        self.k_area = float(self.get_parameter("k_area").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)

        self.get_logger().info("Descargando/Cargando modelo YOLOv8 Nano pre-entrenado...")
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            self.get_logger().error(f"Error cargando YOLO: {e}")
            self.model = None

        self._fallos_consecutivos = 0
        self._frame_count = 0
        self._ultima_caja = None
        self._ultimo_frame = None
        self._lock = threading.Lock()
        self._captura = self._abrir_camara()
        
        # Hilo de lectura asincrona
        threading.Thread(target=self._leer_camara_continuamente, daemon=True).start()

        self._timer   = self.create_timer(TIMER_PERIOD, self._publicar_frame)

    # ── Hilo de Lectura ───────────────────────────────────────────────────

    def _leer_camara_continuamente(self) -> None:
        """Lee continuamente de la camara para mantener vacio el buffer V4L2."""
        while rclpy.ok():
            if self._captura is not None and self._captura.isOpened():
                ret, frame = self._captura.read()
                with self._lock:
                    if ret:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        self._ultimo_frame = frame.copy()
                    else:
                        self._ultimo_frame = None

    # ── Apertura / reapertura de camara ───────────────────────────────────

    def _abrir_camara(self) -> cv2.VideoCapture:
        """Intenta abrir la camara. Loguea resultado sin lanzar excepcion."""
        cap = cv2.VideoCapture(DEVICE_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._fallos_consecutivos = 0
            self.get_logger().info(
                f"Camara abierta /dev/video{DEVICE_INDEX} "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ 30 FPS"
            )
        else:
            self.get_logger().error(
                f"No se pudo abrir /dev/video{DEVICE_INDEX}. "
                "Verifica conexion USB y grupo 'video' (sudo usermod -aG video $USER). "
                f"Reintentando cada {MAX_FALLOS_CONSECUTIVOS} ticks ({MAX_FALLOS_CONSECUTIVOS/30:.0f}s)."
            )
        return cap

    # ── Timer callback ────────────────────────────────────────────────────

    def _publicar_frame(self) -> None:
        # Si la camara no esta abierta, intentar reconexion periodica
        if not self._captura.isOpened():
            self._fallos_consecutivos += 1
            if self._fallos_consecutivos >= MAX_FALLOS_CONSECUTIVOS:
                self.get_logger().warn(
                    f"Camara no disponible. Intentando reconexion a /dev/video{DEVICE_INDEX}..."
                )
                self._captura.release()
                self._captura = self._abrir_camara()
            return

        with self._lock:
            frame = self._ultimo_frame.copy() if self._ultimo_frame is not None else None

        if frame is None:
            self._fallos_consecutivos += 1
            if self._fallos_consecutivos == 1:
                # Primer fallo: loguear sin once=True para detectar desconexiones
                self.get_logger().warn(
                    f"Frame no disponible (fallo #{self._fallos_consecutivos}). "
                    f"Reconexion automatica tras {MAX_FALLOS_CONSECUTIVOS} fallos."
                )
            if self._fallos_consecutivos >= MAX_FALLOS_CONSECUTIVOS:
                self.get_logger().warn(
                    f"Demasiados fallos consecutivos ({self._fallos_consecutivos}). "
                    "Forzando reconexion de camara..."
                )
                self._captura.release()
                self._captura = self._abrir_camara()
            return

        # Frame capturado con exito — resetear contador
        self._fallos_consecutivos = 0
        self._frame_count += 1

        # --- FRAME SKIPPING ---
        if self._frame_count % 5 == 1 or self._ultima_caja is None:
            # --- INFERENCIA YOLO ---
            if self.model is not None:
                # Filtrar exclusivamente clase 39 ('bottle' en COCO)
                results = self.model(frame, classes=[39], verbose=False)
            else:
                results = []

            mejor_conf = 0.0
            mejor_box = None

            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue
                    if conf > mejor_conf:
                        mejor_conf = conf
                        mejor_box = box

            if mejor_box is not None:
                # Extraer ROI
                xyxy = mejor_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Validar limites
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Tamaño
                    ancho = x2 - x1
                    alto = y2 - y1
                    area = ancho * alto
                    tamano_str = "grande" if area > 100000 else "chica"
                    
                    # Recorte Central (Core ROI) para ignorar bordes curvos
                    margen_x = int(ancho * 0.20)
                    margen_y = int(alto * 0.20)
                    roi_core = frame[y1+margen_y : y2-margen_y, x1+margen_x : x2-margen_x]
                    
                    if roi_core.size == 0:
                        estado_str = "limpia"
                    else:
                        # Suciedad (Contraste/StdDev y Brillo) sobre núcleo
                        gray_roi = cv2.cvtColor(roi_core, cv2.COLOR_BGR2GRAY)
                        std_dev = np.std(gray_roi)
                        brillo = np.mean(gray_roi)
                        
                        # Umbral compuesto de ruido/contraste y oscuridad para suciedad
                        estado_str = "sucia" if std_dev > 60.0 or brillo < 90.0 else "limpia"
                    
                    resultado = f"{estado_str}_{tamano_str}"
                    
                    msg = String()
                    msg.data = resultado
                    self._pub_analisis.publish(msg)

            # Dibujar cajas de colision (plot devuelve un ndarray BGR)
            if len(results) > 0:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
            
            self._ultima_caja = annotated_frame
        else:
            # Reutilizar el ultimo frame procesado
            annotated_frame = self._ultima_caja

        try:
            msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "camara_logitech_c270"
            self._pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Error publicando frame: {exc}")

    # ── Ciclo de vida ────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._timer.cancel()
        if self._captura.isOpened():
            self._captura.release()
            self.get_logger().info("Camara liberada.")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoCamara()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
