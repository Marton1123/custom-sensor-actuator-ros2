#!/usr/bin/env python3
"""
nodo_camara.py — Nodo ROS 2 de captura de video (Logitech C270)
Publica sensor_msgs/Image en /camara/video_raw a ~30 FPS usando CvBridge.
Dependencias: sudo apt install python3-opencv ros-jazzy-cv-bridge
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

DEVICE_INDEX = 0
TIMER_PERIOD = 1.0 / 30   # ~33 ms → 30 FPS


class NodoCamara(Node):
    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._captura = cv2.VideoCapture(DEVICE_INDEX)

        if self._captura.isOpened():
            self._captura.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.get_logger().info(
                f"Camara abierta /dev/video{DEVICE_INDEX} "
                f"{int(self._captura.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self._captura.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ 30 FPS"
            )
        else:
            self.get_logger().error(
                f"No se pudo abrir /dev/video{DEVICE_INDEX}. "
                "Verifica conexion USB y grupo 'video' (sudo usermod -aG video $USER)."
            )

        self._timer = self.create_timer(TIMER_PERIOD, self._publicar_frame)

    def _publicar_frame(self) -> None:
        if not self._captura.isOpened():
            return
        ret, frame = self._captura.read()
        if not ret or frame is None:
            self.get_logger().warn("Frame no disponible.", once=True)
            return
        try:
            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "camara_logitech_c270"
            self._pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Error publicando frame: {exc}")

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
