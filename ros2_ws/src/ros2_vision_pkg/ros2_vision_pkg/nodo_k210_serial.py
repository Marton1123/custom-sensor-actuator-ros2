#!/usr/bin/env python3
"""Puente UnitV K210 -> ROS 2 mediante un protocolo UART binario."""

import json
import struct
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import serial
except ImportError as exc:
    raise RuntimeError("Falta pyserial: sudo apt install python3-serial") from exc


MAGIC = b"K2V1"
HEADER = struct.Struct("<4sIII")
MAX_METADATA = 16 * 1024
MAX_JPEG = 512 * 1024

QOS_VIDEO = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)


class NodoK210Serial(Node):
    def __init__(self):
        super().__init__("nodo_k210_serial")
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("reconnect_seconds", 2.0)

        self.serial_port = self.get_parameter("serial_port").value
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.reconnect_seconds = float(self.get_parameter("reconnect_seconds").value)

        self._bridge = CvBridge()
        self._pub_img = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._pub_det = self.create_publisher(String, "/k210/detecciones", 10)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while rclpy.ok() and not self._stop.is_set():
            try:
                self.get_logger().info(
                    f"Abriendo UnitV en {self.serial_port} a {self.baudrate} bps"
                )
                with serial.Serial(
                    self.serial_port,
                    self.baudrate,
                    timeout=0.25,
                    exclusive=True,
                ) as port:
                    self._consume(port)
            except (serial.SerialException, OSError) as exc:
                self.get_logger().warn(
                    f"UnitV no disponible: {exc}. Reintentando...",
                    throttle_duration_sec=2.0,
                )
                self._stop.wait(self.reconnect_seconds)

    def _consume(self, port):
        buffer = bytearray()
        while rclpy.ok() and not self._stop.is_set():
            chunk = port.read(port.in_waiting or 1)
            if not chunk:
                continue
            buffer.extend(chunk)

            while True:
                start = buffer.find(MAGIC)
                if start < 0:
                    if len(buffer) > len(MAGIC):
                        del buffer[:-len(MAGIC)]
                    break
                if start:
                    del buffer[:start]
                if len(buffer) < HEADER.size:
                    break

                _, metadata_len, jpeg_len, expected_checksum = HEADER.unpack_from(buffer)
                if metadata_len > MAX_METADATA or jpeg_len > MAX_JPEG:
                    del buffer[0]
                    continue

                packet_len = HEADER.size + metadata_len + jpeg_len
                if len(buffer) < packet_len:
                    break

                payload = bytes(buffer[HEADER.size:packet_len])
                del buffer[:packet_len]
                if (sum(payload) & 0xFFFFFFFF) != expected_checksum:
                    self.get_logger().warn(
                        "Paquete K210 descartado por checksum",
                        throttle_duration_sec=2.0,
                    )
                    continue
                self._publish_packet(payload[:metadata_len], payload[metadata_len:])

    def _publish_packet(self, metadata_raw, jpeg):
        try:
            metadata = json.loads(metadata_raw.decode("utf-8"))
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG no decodificable")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.get_logger().warn(
                f"Paquete K210 inválido: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        stamp = self.get_clock().now().to_msg()
        detection_msg = String()
        detection_msg.data = json.dumps(metadata, separators=(",", ":"))
        self._pub_det.publish(detection_msg)

        image_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = "unitv_ov7740"
        self._pub_img.publish(image_msg)

    def destroy_node(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NodoK210Serial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
