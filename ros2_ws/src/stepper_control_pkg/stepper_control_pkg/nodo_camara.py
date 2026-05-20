#!/usr/bin/env python3
"""
nodo_camara.py — Nodo ROS 2 de captura de video y visión NCNN
Publica sensor_msgs/Image en /camara/video_raw a ~30 FPS usando CvBridge.

Resiliencia:
    Si la camara falla despues de abrirse, intenta reconectarse automaticamente.
"""

import os
import cv2
import ncnn
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
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


# ─────────────────────────────────────────────
# Helpers: letterbox y NMS (NCNN puro)
# ─────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2

    img_lb = np.full((target, target, 3), 114, dtype=np.uint8)
    img_lb[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

    return img_lb, scale, pad_w, pad_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        order = order[1:][iou < iou_thr]

    return keep


def parse_yolov8_output(raw: np.ndarray, conf_thr: float, iou_thr: float, scale: float, pad_w: int, pad_h: int) -> list[dict]:
    data = np.array(raw)
    if data.ndim == 3:
        data = data[0]

    data = data.T
    confs = data[:, 4]
    mask  = confs >= conf_thr
    if not np.any(mask):
        return []

    filtered = data[mask]
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    scores = filtered[:, 4]

    x1 = (cx - w / 2 - pad_w) / scale
    y1 = (cy - h / 2 - pad_h) / scale
    x2 = (cx + w / 2 - pad_w) / scale
    y2 = (cy + h / 2 - pad_h) / scale

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes_xyxy, scores, iou_thr)

    detections = []
    for i in keep:
        detections.append({
            "x1":   float(boxes_xyxy[i, 0]),
            "y1":   float(boxes_xyxy[i, 1]),
            "x2":   float(boxes_xyxy[i, 2]),
            "y2":   float(boxes_xyxy[i, 3]),
            "conf": float(scores[i]),
        })
    return detections


class NodoCamara(Node):
    def __init__(self) -> None:
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._pub_analisis = self.create_publisher(String, "/analisis_botella", 10)
        self._pub_tamano = self.create_publisher(Float32, "/tamano_estimado", 10)

        # Parametros de IA (NCNN)
        default_model_path = os.path.expanduser("~/custom-sensor-actuator-ros2/IA/models/botellas_ncnn_model")
        self.declare_parameter("modelo_ncnn", default_model_path)
        self.declare_parameter("k_area", 0.05)
        self.declare_parameter("conf_threshold", 0.70)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("ncnn_threads", 2)
        self.declare_parameter("ncnn_input_size", 640)
        self.declare_parameter("distancia_camara_cm", 60.0)
        
        self.modelo_path = str(self.get_parameter("modelo_ncnn").value)
        self.k_area = float(self.get_parameter("k_area").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.ncnn_threads = int(self.get_parameter("ncnn_threads").value)
        self.ncnn_input_size = int(self.get_parameter("ncnn_input_size").value)
        self.distancia_camara_cm = float(self.get_parameter("distancia_camara_cm").value)

        if not os.path.exists(self.modelo_path):
            self.get_logger().error(f"La carpeta del modelo NCNN no existe en la ruta: {self.modelo_path}")
            self.net = None
        else:
            self.get_logger().info(f"Cargando modelo NCNN nativo: {self.modelo_path}")
            try:
                param_path = os.path.join(self.modelo_path, "model.ncnn.param")
                bin_path   = os.path.join(self.modelo_path, "model.ncnn.bin")
                self.net = ncnn.Net()
                self.net.opt.use_vulkan_compute = False
                self.net.opt.num_threads = self.ncnn_threads
                self.net.load_param(param_path)
                self.net.load_model(bin_path)
            except Exception as e:
                self.get_logger().error(f"Error cargando NCNN: {e}")
                self.net = None

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
                "Verifica conexion USB."
            )
        return cap

    # ── Timer callback ────────────────────────────────────────────────────

    def _inferir(self, frame_bgr: np.ndarray) -> list[dict]:
        """Preprocesa, corre inferencia NCNN nativa y decodifica detecciones."""
        if self.net is None:
            return []

        img_lb, scale, pad_w, pad_h = letterbox(frame_bgr, self.ncnn_input_size)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_f32 = img_rgb.astype(np.float32) / 255.0
        img_chw = np.ascontiguousarray(img_f32.transpose(2, 0, 1))
        mat_in = ncnn.Mat(img_chw)
        
        with self.net.create_extractor() as ex:
            ex.set_light_mode(True)
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")
        
        if ret == 0:
            raw = np.array(mat_out)
            return parse_yolov8_output(raw, self.conf_threshold, self.iou_threshold, scale, pad_w, pad_h)
        return []

    def _publicar_frame(self) -> None:
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
                self.get_logger().warn(f"Frame no disponible (fallo #{self._fallos_consecutivos}).")
            if self._fallos_consecutivos >= MAX_FALLOS_CONSECUTIVOS:
                self._captura.release()
                self._captura = self._abrir_camara()
            return

        self._fallos_consecutivos = 0
        self._frame_count += 1

        # --- FRAME SKIPPING e INFERENCIA NCNN ---
        if self._frame_count % 5 == 1 or self._ultima_caja is None:
            detections = self._inferir(frame)
            mejor_conf = 0.0
            mejor_box = None

            for d in detections:
                if d["conf"] > mejor_conf:
                    mejor_conf = d["conf"]
                    mejor_box = d

            resultado = "vacio"
            annotated_frame = frame.copy()

            if mejor_box is not None:
                x1, y1 = int(mejor_box["x1"]), int(mejor_box["y1"])
                x2, y2 = int(mejor_box["x2"]), int(mejor_box["y2"])
                ancho = x2 - x1
                alto = y2 - y1
                area = ancho * alto
                resultado = "grande" if area > 33000 else "chica"

                # Publicar área física estimada (Lo mejor de nodo_vision)
                msg_tam = Float32()
                msg_tam.data = float(area * self.k_area)
                self._pub_tamano.publish(msg_tam)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{resultado} {mejor_conf:.2f}", (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            msg_str = String()
            msg_str.data = resultado
            self._pub_analisis.publish(msg_str)
            
            self._ultima_caja = annotated_frame
        else:
            annotated_frame = self._ultima_caja

        try:
            msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
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
        if hasattr(self, "net"):
            self.net = None
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
