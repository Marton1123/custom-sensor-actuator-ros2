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

import os
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
        default_model_path = os.path.expanduser("~/custom-sensor-actuator-ros2/IA/models/botellas_ncnn_model")
        self.declare_parameter("modelo_ncnn", default_model_path)
        self.declare_parameter("k_area", 0.05)
        self.declare_parameter("conf_threshold", 0.70)
        
        self.modelo_path = str(self.get_parameter("modelo_ncnn").value)
        self.k_area = float(self.get_parameter("k_area").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)

        if not os.path.exists(self.modelo_path):
            self.get_logger().error(f"La carpeta del modelo NCNN no existe en la ruta: {self.modelo_path}")
            self.model = None
        else:
            self.get_logger().info(f"Cargando modelo NCNN: {self.modelo_path}")
            try:
                self.model = YOLO(self.modelo_path, task="detect")
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
                # Modelo entrenado para detectar solo botellas (no requiere filtrar por clase COCO)
                results = self.model(frame, verbose=False)
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

            resultado = "vacio"
            if mejor_box is not None:
                xyxy = mejor_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                ancho = x2 - x1
                alto = y2 - y1
                area = ancho * alto
                resultado = "grande" if area > 33000 else "chica"

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
