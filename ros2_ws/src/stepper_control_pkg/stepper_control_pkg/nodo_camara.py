#!/usr/bin/env python3
"""
nodo_camara.py  Nodo ROS 2 de Captura de Video

Captura fotogramas desde una cámara UVC (V4L2) en un hilo daemon y
publica los fotogramas crudos (raw) en el tópico /camara/video_raw.
La inferencia ahora delegada a nodo_vision.

Resiliencia:
    Si la camara falla despues de abrirse, intenta reconectarse automaticamente.
"""

import os
import cv2
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

QOS_VIDEO = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

TIMER_PERIOD         = 1.0 / 30   # ~33 ms ? 30 FPS
MAX_FALLOS_CONSECUTIVOS = 30      # ~1 segundo de fallos antes de reconectar


class NodoCamara(Node):
    """Nodo ROS 2 de captura de video para la Raspberry Pi 5.

    Publica:
        /camara/video_raw (sensor_msgs/Image): Fotograma crudo original.

    Parámetros YAML (nodo_camara):
        device_index (int/str): Índice o ruta de la cámara. Default: 0
    """

    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        
        self.declare_parameter("device_index", 0)
        self.device_index = self.get_parameter("device_index").value

        self._fallos_consecutivos = 0
        self._ultimo_frame = None
        self._lock = threading.Lock()
        self._captura = self._abrir_camara()
        
        # Hilo de lectura asincrona
        threading.Thread(target=self._leer_camara_continuamente, daemon=True).start()

        self._timer   = self.create_timer(TIMER_PERIOD, self._publicar_frame)

    # -- Hilo de Lectura ---------------------------------------------------

    def _leer_camara_continuamente(self) -> None:
        """Lee continuamente de la camara para mantener vacio el buffer V4L2."""
        while rclpy.ok():
            if self._captura is not None and self._captura.isOpened():
                ret, frame = self._captura.read()
                with self._lock:
                    if ret:
                        # Si tu cámara requiere rotación por su montura física, hazlo aquí.
                        # Por defecto dejaremos sin rotación. Si necesitas rotar 90 grados:
                        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        self._ultimo_frame = frame.copy()
                    else:
                        self._ultimo_frame = None

    # -- Apertura / reapertura de camara -----------------------------------

    def _abrir_camara(self) -> cv2.VideoCapture:
        """Intenta abrir la camara iterando sobre puertos 0-5 para evitar nodos de metadatos V4L2."""
        import time
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                for _ in range(3):  # Give the camera a moment to warm up
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self._fallos_consecutivos = 0
                        
                        real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        self.get_logger().info(f"Cámara conectada exitosamente en el puerto /dev/video{i} ({int(real_w)}x{int(real_h)})")
                        self.device_index = i
                        return cap
                    time.sleep(0.1)
                cap.release()
            else:
                cap.release()
                
        self.get_logger().error("No se encontró ninguna cámara de captura de video válida en los puertos 0-5")
        return None

    def _reconectar_camara(self) -> None:
        """Cierra el handle actual y reabre usando _abrir_camara()."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
        self._captura = self._abrir_camara()

    # -- Ciclo Principal (Timer a 30 FPS) ----------------------------------

    def _publicar_frame(self) -> None:
        """Extrae el fotograma del hilo daemon y lo publica localmente."""
        with self._lock:
            frame = self._ultimo_frame
            self._ultimo_frame = None

        if frame is None:
            self._fallos_consecutivos += 1
            if self._fallos_consecutivos % 10 == 0:
                self.get_logger().warn(
                    f"Camara no disponible. Intentando reconexion (fallos: {self._fallos_consecutivos})..."
                )
            if self._fallos_consecutivos > MAX_FALLOS_CONSECUTIVOS:
                self._reconectar_camara()
            return

        self._fallos_consecutivos = 0

        # Publicar crudo a /camara/video_raw para que nodo_vision procese
        msg_img = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self._pub.publish(msg_img)


def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoCamara()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.get_logger().info("Cancelado por KeyboardInterrupt.")
    finally:
        if nodo._captura:
            nodo._captura.release()
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
