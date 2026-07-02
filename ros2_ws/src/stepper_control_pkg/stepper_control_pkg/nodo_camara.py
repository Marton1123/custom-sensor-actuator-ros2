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
        self._captura = None
        self._running = True
        self._lock = threading.Lock()
        
        # Iniciar cámara por primera vez
        self._captura = self._abrir_camara()
        
        # Hilo de lectura y publicación asíncrona
        self._thread = threading.Thread(target=self._leer_y_publicar_continuamente, daemon=True)
        self._thread.start()

    # -- Apertura / reapertura de camara -----------------------------------

    def _abrir_camara(self) -> cv2.VideoCapture:
        """Intenta abrir la cámara en el índice configurado (device_index), reintentando si está ocupado por procesos anteriores."""
        import time
        device_path = f'/dev/video{self.device_index}'
        
        for intento in range(5):
            if not os.path.exists(device_path):
                self.get_logger().warn(f"El dispositivo {device_path} no existe en el sistema. Reintentando...")
                time.sleep(1.0)
                continue
                
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                # Configurar propiedades antes de intentar leer
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                time.sleep(0.3)  # Espera para estabilizar el sensor físico
                ret, frame = cap.read()
                if ret and frame is not None:
                    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    self.get_logger().info(f"Cámara conectada exitosamente en {device_path} ({int(real_w)}x{int(real_h)})")
                    self._fallos_consecutivos = 0
                    return cap
                cap.release()
            else:
                cap.release()
                
            self.get_logger().warn(f"El puerto {device_path} está ocupado o inicializándose (intento {intento+1}/5). Esperando 1s...")
            time.sleep(1.0)
            
        self.get_logger().error(f"No se pudo abrir la cámara en {device_path} tras 5 intentos.")
        return None

    def _reconectar_camara(self) -> None:
        """Cierra el handle actual y reabre usando _abrir_camara()."""
        with self._lock:
            if self._captura is not None:
                self._captura.release()
                self._captura = None
            self._captura = self._abrir_camara()

    # -- Hilo Principal de Lectura y Publicación Directa -------------------

    def _leer_y_publicar_continuamente(self) -> None:
        """Lee de la cámara y publica directamente para minimizar latencia y consumo de CPU."""
        import time
        while rclpy.ok() and self._running:
            with self._lock:
                cap = self._captura
            
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._fallos_consecutivos = 0
                    
                    # Rotar si es necesario (montura física de 90 grados)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                    try:
                        msg_img = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                        self._pub.publish(msg_img)
                    except Exception as e:
                        self.get_logger().error(f"Error al publicar frame: {e}")
                else:
                    self._fallos_consecutivos += 1
                    if self._fallos_consecutivos % 10 == 0:
                        self.get_logger().warn(
                            f"Fallo al leer frame ({self._fallos_consecutivos}/{MAX_FALLOS_CONSECUTIVOS})"
                        )
                    
                    if self._fallos_consecutivos >= MAX_FALLOS_CONSECUTIVOS:
                        self.get_logger().error("Demasiados fallos de lectura consecutivos. Reconectando cámara...")
                        self._reconectar_camara()
                    time.sleep(0.05)  # Evitar bucle sin fin consumiendo CPU
            else:
                time.sleep(1.0)
                self._reconectar_camara()


def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoCamara()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.get_logger().info("Cancelado por KeyboardInterrupt.")
    finally:
        nodo._running = False
        with nodo._lock:
            if nodo._captura:
                nodo._captura.release()
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
