#!/usr/bin/env python3
"""
nodo_vision.py — Nodo ROS 2 fusionado de captura + inferencia YOLOv8n NCNN.

Captura directa V4L2 (absorbe nodo_camara), inferencia CPU-only con ncnn-python.
Pre/post procesamiento en NumPy puro: letterbox + NMS manual.
Sin PyTorch, sin Ultralytics, sin CvBridge.

Publica:
  /botella_detectada  (std_msgs/Bool)   — True si confianza > conf_threshold
  /tamano_estimado    (std_msgs/Float32) — Área estimada en cm²

Dependencias RPi:
  sudo apt install python3-opencv ros-jazzy-rclpy ros-jazzy-std-msgs
  pip install ncnn  (en el venv del paquete)
"""

import os
import cv2
import ncnn
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


# ─────────────────────────────────────────────
# Helpers: letterbox y NMS (sin dependencias ML)
# ─────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona manteniendo aspecto y rellena con gris (114).

    Returns:
        img_lb  : imagen 640×640 lista para inferencia
        scale   : factor de escala aplicado
        pad_w   : píxeles de padding horizontal (cada lado)
        pad_h   : píxeles de padding vertical   (cada lado)
    """
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
    """Non-Maximum Suppression clásico sobre arrays [N,4] xyxy y [N] scores."""
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


def parse_yolov8_output(
    raw: np.ndarray,
    conf_thr: float,
    iou_thr: float,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> list[dict]:
    """
    Decodifica la salida cruda de YOLOv8n exportado a NCNN.

    La cabeza de detección YOLOv8 exportada produce un tensor con shape:
      [1, 5, 8400]  →  (batch, cx cy w h conf, anchors)  para 1 clase
    o equivalentemente raw shape (5, 8400) tras quitar el batch.

    Returns:
        Lista de dicts con claves: x1 y1 x2 y2 conf (coords en píxeles originales)
    """
    # raw puede llegar como (1,5,8400) o (5,8400)
    data = np.array(raw)
    if data.ndim == 3:
        data = data[0]          # → (5, 8400)

    # Transponer a (8400, 5): cada fila = [cx, cy, w, h, conf]
    data = data.T               # → (8400, 5)

    confs = data[:, 4]
    mask  = confs >= conf_thr
    if not np.any(mask):
        return []

    filtered = data[mask]       # (M, 5)
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    scores = filtered[:, 4]

    # cx,cy,w,h están en espacio letterbox 640×640 → convertir a xyxy en imagen original
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


# ─────────────────────────────────────────────
# Nodo principal
# ─────────────────────────────────────────────

class NodoVision(Node):

    # Nombre del paquete ROS 2 donde viven los modelos
    _PKG_NAME = "custom_sensor_actuator"

    def __init__(self) -> None:
        super().__init__("nodo_vision")

        # ── Parámetros ──────────────────────────────────────────────────────────
        # Ruta relativa dentro de <paquete>/models/  (sin extensión)
        self.declare_parameter("modelo_relativo", "botellas_ncnn_model/model.ncnn")
        self.declare_parameter("cam_id",          0)
        self.declare_parameter("cam_width",       640)
        self.declare_parameter("cam_height",      480)
        self.declare_parameter("cam_fps",         30)
        self.declare_parameter("infer_hz",        10.0)   # timer de inferencia
        self.declare_parameter("k_area",          0.05)   # cm² / px²
        self.declare_parameter("conf_threshold",  0.70)
        self.declare_parameter("iou_threshold",   0.45)
        self.declare_parameter("ncnn_threads",    2)      # conservador sin refrigeración
        self.declare_parameter("ncnn_input_size", 640)

        modelo_rel    = str(self.get_parameter("modelo_relativo").value)
        self.cam_id   = int(self.get_parameter("cam_id").value)
        cam_w         = int(self.get_parameter("cam_width").value)
        cam_h         = int(self.get_parameter("cam_height").value)
        cam_fps       = int(self.get_parameter("cam_fps").value)
        infer_hz      = float(self.get_parameter("infer_hz").value)
        self.k_area   = float(self.get_parameter("k_area").value)
        self.conf_thr = float(self.get_parameter("conf_threshold").value)
        self.iou_thr  = float(self.get_parameter("iou_threshold").value)
        threads       = int(self.get_parameter("ncnn_threads").value)
        self.inp_size = int(self.get_parameter("ncnn_input_size").value)

        # ── Rutas del modelo (relativas al share del paquete) ───────────────────
        try:
            pkg_share = get_package_share_directory(self._PKG_NAME)
        except Exception:
            # Fallback: directorio del propio script (útil en desarrollo)
            pkg_share = os.path.dirname(os.path.abspath(__file__))
            self.get_logger().warning(
                f"Paquete '{self._PKG_NAME}' no encontrado via ament_index. "
                f"Usando fallback: {pkg_share}"
            )

        base_path  = os.path.join(pkg_share, "models", modelo_rel)
        param_path = base_path + ".param"
        bin_path   = base_path + ".bin"

        for p in (param_path, bin_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Archivo de modelo no encontrado: {p}\n"
                    "Verifica 'modelo_relativo' o copia el modelo a "
                    f"{os.path.join(pkg_share, 'models')}"
                )

        # ── Carga del modelo NCNN ───────────────────────────────────────────────
        self.get_logger().info(f"Cargando modelo NCNN: {base_path}.*")
        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = False   # CPU-only (RPi 5 no tiene Vulkan útil)
        self._net.opt.num_threads        = threads
        self._net.load_param(param_path)
        self._net.load_model(bin_path)
        self.get_logger().info(f"Modelo cargado | threads={threads}")

        # ── Apertura de cámara (V4L2) ───────────────────────────────────────────
        self._cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            self._cap.set(cv2.CAP_PROP_FPS,          cam_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimiza latencia
            real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(
                f"Cámara /dev/video{self.cam_id} abierta: {real_w}x{real_h}@{cam_fps}"
            )
        else:
            raise RuntimeError(
                f"No se pudo abrir /dev/video{self.cam_id}. "
                "Verifica conexión USB y permisos: sudo usermod -aG video $USER"
            )

        # ── Publicadores ────────────────────────────────────────────────────────
        self._pub_det = self.create_publisher(Bool,    "/botella_detectada", 10)
        self._pub_tam = self.create_publisher(Float32, "/tamano_estimado",   10)

        # ── Timer de inferencia ─────────────────────────────────────────────────
        periodo = 1.0 / infer_hz if infer_hz > 0.0 else 0.1
        self._timer = self.create_timer(periodo, self._inferencia_callback)

        self.get_logger().info(
            f"nodo_vision listo | infer_hz={infer_hz:.1f} "
            f"| conf>{self.conf_thr:.2f} | K_area={self.k_area}"
        )

    # ── Callback principal ───────────────────────────────────────────────────────

    def _inferencia_callback(self) -> None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.get_logger().warning("Frame no disponible.", once=True)
            return

        detecciones = self._inferir(frame)

        # Seleccionar detección con mayor confianza
        detectado  = False
        tamano_cm2 = 0.0

        if detecciones:
            mejor = max(detecciones, key=lambda d: d["conf"])
            w_px  = mejor["x2"] - mejor["x1"]
            h_px  = mejor["y2"] - mejor["y1"]
            area_px = w_px * h_px
            detectado  = True
            tamano_cm2 = area_px * self.k_area

        # Publicar siempre /botella_detectada
        msg_det      = Bool()
        msg_det.data = detectado
        self._pub_det.publish(msg_det)

        # Publicar /tamano_estimado solo si hay detección
        if detectado:
            msg_tam      = Float32()
            msg_tam.data = float(tamano_cm2)
            self._pub_tam.publish(msg_tam)

    # ── Inferencia NCNN ──────────────────────────────────────────────────────────

    def _inferir(self, frame_bgr: np.ndarray) -> list[dict]:
        """Preprocesa, corre inferencia NCNN y decodifica detecciones."""

        # 1. Letterbox → 640×640
        img_lb, scale, pad_w, pad_h = letterbox(frame_bgr, self.inp_size)

        # 2. BGR → RGB, float32 normalizado [0,1]
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_f32 = img_rgb.astype(np.float32) / 255.0

        # 3. HWC → CHW (requerido por NCNN)
        img_chw = np.ascontiguousarray(img_f32.transpose(2, 0, 1))

        # 4. Crear Mat NCNN y correr extractor
        mat_in = ncnn.Mat(img_chw)

        with self._net.create_extractor() as ex:
            ex.set_light_mode(True)   # libera tensores intermedios → ahorra RAM
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")

        if ret != 0:
            self.get_logger().error("Error en ncnn extract()")
            return []

        # 5. Decodificar salida
        raw = np.array(mat_out)
        return parse_yolov8_output(raw, self.conf_thr, self.iou_thr, scale, pad_w, pad_h)

    # ── Limpieza ─────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.cancel()
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
            self.get_logger().info("Cámara liberada.")
        if hasattr(self, "_net"):
            self._net = None
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