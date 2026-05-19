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

DEVICE_INDEX         = 0
TIMER_PERIOD         = 1.0 / 30   # ~33 ms → 30 FPS
MAX_FALLOS_CONSECUTIVOS = 30      # ~1 segundo de fallos antes de reconectar


class NodoCamara(Node):
    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._fallos_consecutivos = 0
        self._captura = self._abrir_camara()
        self._timer   = self.create_timer(TIMER_PERIOD, self._publicar_frame)

    # ── Apertura / reapertura de camara ───────────────────────────────────

    def _abrir_camara(self) -> cv2.VideoCapture:
        """Intenta abrir la camara. Loguea resultado sin lanzar excepcion."""
        cap = cv2.VideoCapture(DEVICE_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

        ret, frame = self._captura.read()
        if not ret or frame is None:
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

        try:
            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
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
