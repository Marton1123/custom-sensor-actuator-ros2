#!/usr/bin/env python3
"""Nodo de visión por CPU usando modelo YOLOv8 exportado a NCNN."""

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from ultralytics import YOLO


class NodoVision(Node):
    def __init__(self) -> None:
        super().__init__("nodo_vision")

        self.declare_parameter("modelo_ncnn", "/home/pi/modelos/best_ncnn_model")
        self.declare_parameter("cam_id", 0)
        self.declare_parameter("cam_width", 640)
        self.declare_parameter("cam_height", 480)
        self.declare_parameter("cam_fps", 30)
        self.declare_parameter("infer_hz", 10.0)
        self.declare_parameter("k_area", 0.05)
        self.declare_parameter("conf_threshold", 0.70)

        self.modelo_path = str(self.get_parameter("modelo_ncnn").value)
        self.cam_id = int(self.get_parameter("cam_id").value)
        self.cam_width = int(self.get_parameter("cam_width").value)
        self.cam_height = int(self.get_parameter("cam_height").value)
        self.cam_fps = int(self.get_parameter("cam_fps").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)
        self.k_area = float(self.get_parameter("k_area").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)

        self.pub_deteccion = self.create_publisher(Bool, "/botella_detectada", 10)
        self.pub_tamano = self.create_publisher(Float32, "/tamano_estimado", 10)

        self.get_logger().info(f"Cargando modelo NCNN: {self.modelo_path}")
        self.model = YOLO(self.modelo_path, task="detect")

        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.cam_id}")

        periodo = 1.0 / self.infer_hz if self.infer_hz > 0.0 else 0.1
        self.timer = self.create_timer(periodo, self.timer_callback)
        self.get_logger().info(
            f"nodo_vision listo | cam={self.cam_id} {self.cam_width}x{self.cam_height}@{self.cam_fps} "
            f"| infer_hz={self.infer_hz:.1f} | conf>{self.conf_threshold:.2f} | K_area={self.k_area}"
        )

    def timer_callback(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("No se pudo capturar frame de cámara.")
            return

        results = self.model(frame, verbose=False)

        detectado = False
        tamano = 0.0
        mejor_conf = 0.0
        mejor_area_pix = 0.0

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                w = float(box.xywh[0][2])
                h = float(box.xywh[0][3])
                area_pix = w * h

                if conf > mejor_conf:
                    mejor_conf = conf
                    mejor_area_pix = area_pix

        if mejor_conf >= self.conf_threshold:
            detectado = True
            tamano = mejor_area_pix * self.k_area

        msg_det = Bool()
        msg_det.data = detectado
        self.pub_deteccion.publish(msg_det)

        if detectado:
            msg_tam = Float32()
            msg_tam.data = float(tamano)
            self.pub_tamano.publish(msg_tam)

    def destroy_node(self) -> None:
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    nodo = NodoVision()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
