#!/usr/bin/env python3
"""
nodo_camara.py — Nodo ROS 2 de Captura de Video

Captura fotogramas desde una cámara UVC (V4L2) en un hilo daemon.
Soporta cámaras estereoscópicas (ej. Stereolabs ZED en modo UVC side-by-side)
extrayendo la vista del ojo izquierdo, aplicando rotación si la montura física
lo requiere y redimensionando a la resolución esperada por el pipeline de visión.
Publica los fotogramas procesados en el tópico /camara/video_raw.

Resiliencia:
    Si la cámara falla después de abrirse, intenta reconectarse automáticamente.
"""

import os
import time
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
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

MAX_FALLOS_CONSECUTIVOS: int = 30


class NodoCamara(Node):
    """Nodo ROS 2 de captura de video para Raspberry Pi 5 / Linux.

    Publica:
        /camara/video_raw (sensor_msgs/Image): Fotograma recortado y acondicionado.

    Parámetros YAML / ROS 2:
        device_index (int): Índice o identificador del dispositivo V4L2 (/dev/videoX). Default: 0
        recortar_ojo_izquierdo (bool): Extrae la mitad izquierda de un frame estereoscópico side-by-side. Default: True
        redimensionar (bool): Habilita el escalado a resolución estándar. Default: True
        ancho_objetivo (int): Ancho de salida en píxeles. Default: 640
        alto_objetivo (int): Alto de salida en píxeles. Default: 480
        rotar_90_horario (bool): Rota 90 grados en sentido horario según orientación física. Default: True
        ancho_captura (int): Ancho solicitado a V4L2 (0 = resolución nativa del dispositivo). Default: 0
        alto_captura (int): Alto solicitado a V4L2 (0 = resolución nativa del dispositivo). Default: 0
    """

    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)

        self.declare_parameter("device_index", 0)
        self.declare_parameter("recortar_ojo_izquierdo", True)
        self.declare_parameter("redimensionar", True)
        self.declare_parameter("ancho_objetivo", 640)
        self.declare_parameter("alto_objetivo", 480)
        self.declare_parameter("rotar_90_horario", True)
        self.declare_parameter("ancho_captura", 0)
        self.declare_parameter("alto_captura", 0)

        self.device_index: int = int(self.get_parameter("device_index").value)
        self.recortar_ojo_izquierdo: bool = bool(self.get_parameter("recortar_ojo_izquierdo").value)
        self.redimensionar: bool = bool(self.get_parameter("redimensionar").value)
        self.ancho_objetivo: int = int(self.get_parameter("ancho_objetivo").value)
        self.alto_objetivo: int = int(self.get_parameter("alto_objetivo").value)
        self.rotar_90_horario: bool = bool(self.get_parameter("rotar_90_horario").value)
        self.ancho_captura: int = int(self.get_parameter("ancho_captura").value)
        self.alto_captura: int = int(self.get_parameter("alto_captura").value)

        self._fallos_consecutivos: int = 0
        self._captura: Optional[cv2.VideoCapture] = None
        self._running: bool = True
        self._lock: threading.Lock = threading.Lock()

        self._captura = self._abrir_camara()

        self._thread: threading.Thread = threading.Thread(
            target=self._leer_y_publicar_continuamente,
            daemon=True
        )
        self._thread.start()

    # -- Apertura / reapertura de camara -----------------------------------

    def _abrir_camara(self) -> Optional[cv2.VideoCapture]:
        """Abre la cámara en el índice configurado, reintentando si el dispositivo está ocupado."""
        device_path = f"/dev/video{self.device_index}"

        for intento in range(5):
            if not os.path.exists(device_path):
                self.get_logger().warn(
                    f"El dispositivo {device_path} no existe en el sistema. Reintentando..."
                )
                time.sleep(1.0)
                continue

            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                if self.ancho_captura > 0 and self.alto_captura > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho_captura)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto_captura)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                time.sleep(0.3)
                ret, frame = cap.read()
                if ret and frame is not None:
                    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.get_logger().info(
                        f"Cámara conectada exitosamente en {device_path} ({real_w}x{real_h})"
                    )
                    self._fallos_consecutivos = 0
                    return cap
                cap.release()
            else:
                cap.release()

            self.get_logger().warn(
                f"El puerto {device_path} está ocupado o inicializándose (intento {intento + 1}/5). Esperando 1s..."
            )
            time.sleep(1.0)

        self.get_logger().error(f"No se pudo abrir la cámara en {device_path} tras 5 intentos.")
        return None

    def _reconectar_camara(self) -> None:
        """Cierra el handle actual y reabre la cámara de forma segura."""
        with self._lock:
            if self._captura is not None:
                self._captura.release()
                self._captura = None
            self._captura = self._abrir_camara()

    # -- Hilo Principal de Lectura y Publicación Directa -------------------

    def _leer_y_publicar_continuamente(self) -> None:
        """Lee el flujo de video, aplica recorte estereoscópico, rotación y reescalado antes de publicar."""
        while rclpy.ok() and self._running:
            with self._lock:
                cap = self._captura

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._fallos_consecutivos = 0

                    if self.recortar_ojo_izquierdo:
                        alto, ancho = frame.shape[:2]
                        frame = frame[:, :ancho // 2]

                    if self.rotar_90_horario:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    if self.redimensionar:
                        alto_act, ancho_act = frame.shape[:2]
                        if ancho_act != self.ancho_objetivo or alto_act != self.alto_objetivo:
                            frame = cv2.resize(
                                frame,
                                (self.ancho_objetivo, self.alto_objetivo),
                                interpolation=cv2.INTER_LINEAR,
                            )

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
                        self.get_logger().error(
                            "Demasiados fallos de lectura consecutivos. Reconectando cámara..."
                        )
                        self._reconectar_camara()
                    time.sleep(0.05)
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
