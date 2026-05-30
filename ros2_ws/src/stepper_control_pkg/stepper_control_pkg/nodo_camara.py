#!/usr/bin/env python3
"""
nodo_camara.py — Nodo ROS 2 de Visión Autónoma e Inferencia de Objetos.

Captura fotogramas desde una cámara UVC (V4L2) en un hilo daemon y ejecuta
inferencia YOLOv8 mediante el motor NCNN nativo para detectar y clasificar
elementos en escena. Aplica análisis HSV sobre la ROI para determinar el
estado del objetivo (OPTIMO / ANOMALIA) y publica los resultados en el bus
ROS 2 a ~30 FPS usando CvBridge.

Resiliencia:
    Si la camara falla despues de abrirse, intenta reconectarse automaticamente.
"""

import os
import cv2
import time
import ncnn
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool
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
    """Nodo ROS 2 de visión autónoma para detección y clasificación de objetivos.

    Integra captura de video UVC, inferencia NCNN (YOLOv8) y análisis de
    anomalías HSV en un ciclo de máquina de estados asíncrona.

    Publica:
        /camara/video_raw (sensor_msgs/Image): Fotograma anotado en tiempo real.
        /camara/video_segmentado (sensor_msgs/Image): Mapa de anomalías HSV.
        /clasificacion_objeto (std_msgs/String): Veredicto de clasificación.
        /tamano_estimado (std_msgs/Float32): Área transversal estimada en cm².
        /comando_grados (std_msgs/Float32): Ángulo de actuación (90.0 aceptación, -90.0 rechazo, 0.0 home).

    Parámetros YAML (nodo_camara):
        distancia_camara_cm (float): Distancia de referencia en cm. Default: 60.0.
        modelo_ncnn (str): Ruta al directorio del modelo NCNN.
        k_area (float): Factor escala píxel→cm². Default: 0.05.
        conf_threshold (float): Umbral de confianza YOLO. Default: 0.70.
        iou_threshold (float): Umbral IoU para NMS. Default: 0.45.
        ncnn_threads (int): Hilos de inferencia NCNN. Default: 2.
        ncnn_input_size (int): Resolución cuadrada de entrada. Default: 640.
    """

    def __init__(self) -> None:
        """Inicializa publishers, parámetros YAML, modelo NCNN y hilo de captura.

        Publishers creados:
            - /camara/video_raw, /camara/video_segmentado, /camara/foto_anotada
            - /clasificacion_objeto, /tamano_estimado
            - /comando_grados: control del mecanismo de actuación físico.
        """
        super().__init__("nodo_camara")
        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Image, "/camara/video_raw", QOS_VIDEO)
        self._pub_analisis = self.create_publisher(String, "/clasificacion_objeto", 10)
        self._pub_tamano = self.create_publisher(Float32, "/tamano_estimado", 10)
        self._pub_video_segmentado = self.create_publisher(Image, "/camara/video_segmentado", QOS_VIDEO)
        self._pub_foto_anotada = self.create_publisher(Image, "/camara/foto_anotada", QOS_VIDEO)
        self._pub_comando_grados = self.create_publisher(Float32, "/comando_grados", 10)
        
        self._estado_actual     = 'BUSQUEDA'
        self._frames_botella    = 0   # Contador de frames con deteccion estable
        self._frames_analizando = 0   # Contador de frames en estado ANALIZANDO
        self._frames_vacio      = 0   # Contador de frames sin deteccion en ESPERA_RETIRO
        self._ultimo_veredicto  = ""
        self._frame_congelado   = None   # Copia estatica del frame al alcanzar umbral
        self._bbox_congelado    = None   # BBox correspondiente al frame congelado
        self._canvas_veredicto  = None   # Imagen HSV resultado de _aplicar_segmentacion
        self._ultimo_segmentado = None

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

    def _aplicar_segmentacion(self, frame_bgr: np.ndarray, bbox: dict) -> tuple:
        """Aplica filtros HSV sobre la ROI del objetivo para clasificar su estado.

        Combina una máscara de valor bajo (opacidad/oscuridad) con una máscara
        de saturación alta (líquidos/etiquetas) para cuantificar el porcentaje
        de anomalía. Genera una imagen de visualización con las anomalías en rojo.

        Args:
            frame_bgr: Fotograma fuente en espacio de color BGR.
            bbox: Diccionario con claves x1, y1, x2, y2 de la caja delimitadora.

        Returns:
            tuple: (canvas_visualizacion, estado) donde estado es 'OPTIMO' o 'ANOMALIA'.
        """
        frame_copia = frame_bgr.copy()
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_copia.shape[1], x2)
        y2 = min(frame_copia.shape[0], y2)

        if (x2 <= x1) or (y2 <= y1):
            return frame_copia, "OPTIMO"

        roi = frame_copia[y1:y2, x1:x2]

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Filtro de opacidad/suciedad (Value bajo)
        mask_dark = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([179, 255, 80]))

        # Filtro de liquidos/etiquetas (Saturacion alta)
        mask_sat = cv2.inRange(roi_hsv, np.array([0, 100, 0]), np.array([179, 255, 255]))

        # Combinar mascaras
        mask_anomalia = cv2.bitwise_or(mask_dark, mask_sat)

        # Calculo de porcentaje de anomalia respecto a ROI
        total_pixels = roi.shape[0] * roi.shape[1]
        anomalia_pixels = np.count_nonzero(mask_anomalia)
        porcentaje = anomalia_pixels / total_pixels if total_pixels > 0 else 0.0

        estado_limpieza = "OPTIMO" if porcentaje < 0.10 else "ANOMALIA"

        # 1. Canvas global con fondo gris oscuro para contraste
        canvas = np.ones_like(frame_bgr) * 50

        # 2. ROI original para aplicar la máscara
        roi = frame_bgr[y1:y2, x1:x2]

        # 3. Colorear anomalías en el ROI (sin tocar el canvas directamente)
        roi_anomalias = np.zeros_like(roi)
        roi_anomalias[mask_anomalia > 0] = [0, 0, 255]  # Rojo puro

        # 4. Inyectar el ROI procesado en las coordenadas exactas del canvas
        canvas[y1:y2, x1:x2] = roi_anomalias

        # 5. BBox verde sobre el canvas global final
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return canvas, estado_limpieza

    def _publicar_frame(self) -> None:
        """Callback de temporizador (~30 FPS). Orquesta la máquina de estados de visión.

        Estados:
            BUSQUEDA      : YOLO activo. Acumula frames con detección. Publica feed
                            en vivo en /video_raw y texto 'ESPERANDO' en /video_segmentado.
            ANALIZANDO    : YOLO inactivo. Congela el frame. Publica frame congelado
                            en /video_raw y texto 'ANALIZANDO' en /video_segmentado
                            durante 15 frames (~0.5 s) antes de ejecutar el filtro HSV.
            ESPERA_RETIRO : YOLO activo para detectar retirada del objeto. Mantiene
                            frame congelado en /video_raw y canvas HSV en /video_segmentado.
            (retorno)     : Tras 30 frames vacios vuelve a BUSQUEDA y resetea todo.
        """
        # ── Guardias de disponibilidad de frame ──────────────────────────
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
        stamp = self.get_clock().now().to_msg()

        # ══════════════════════════════════════════════════════════════════
        # FASE 1: BUSQUEDA — YOLO activo, feed en vivo
        # ══════════════════════════════════════════════════════════════════
        if self._estado_actual == 'BUSQUEDA':
            detections = self._inferir(frame)
            mejor_conf = 0.0
            mejor_box  = None
            for d in detections:
                if d["conf"] > mejor_conf:
                    mejor_conf = d["conf"]
                    mejor_box  = d

            # Canvas fijo inferior: texto ESPERANDO
            canvas_espera = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                canvas_espera, "ESPERANDO ELEMENTO...",
                (55, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA
            )
            try:
                msg_seg = self._bridge.cv2_to_imgmsg(canvas_espera, encoding="bgr8")
                msg_seg.header.stamp = stamp
                msg_seg.header.frame_id = "camara_logitech_c270"
                self._pub_video_segmentado.publish(msg_seg)
            except Exception: pass

            if mejor_box is None or mejor_conf < 0.65:
                # Sin deteccion: feed limpio en vivo
                self._frames_botella = 0
                try:
                    msg_raw = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    msg_raw.header.stamp = stamp
                    msg_raw.header.frame_id = "camara_logitech_c270"
                    self._pub.publish(msg_raw)
                except Exception: pass
                msg_str = String()
                msg_str.data = "vacio"
                self._pub_analisis.publish(msg_str)
                return

            # Deteccion valida: acumular frames
            self._frames_botella += 1
            x1 = int(mejor_box["x1"]); y1 = int(mejor_box["y1"])
            x2 = int(mejor_box["x2"]); y2 = int(mejor_box["y2"])

            # Feed en vivo con BBox
            img_vivo = frame.copy()
            cv2.rectangle(img_vivo, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_vivo, f"CONF: {mejor_conf:.2f}",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            try:
                msg_raw = self._bridge.cv2_to_imgmsg(img_vivo, encoding="bgr8")
                msg_raw.header.stamp = stamp
                msg_raw.header.frame_id = "camara_logitech_c270"
                self._pub.publish(msg_raw)
            except Exception: pass

            msg_str = String()
            msg_str.data = "analizando"
            self._pub_analisis.publish(msg_str)

            # Transicion al estado ANALIZANDO
            if self._frames_botella >= 15:
                self._frame_congelado  = frame.copy()
                self._bbox_congelado   = mejor_box
                self._frames_analizando = 0
                self._estado_actual    = 'ANALIZANDO'
                self.get_logger().info(
                    f"Objeto estabilizado ({self._frames_botella} frames). "
                    "Transicion a ANALIZANDO."
                )
            return

        # ══════════════════════════════════════════════════════════════════
        # FASE 2: ANALIZANDO — delay asincrono por contador de frames
        # ══════════════════════════════════════════════════════════════════
        elif self._estado_actual == 'ANALIZANDO':
            self._frames_analizando += 1

            # Visor superior: frame congelado con BBox
            if self._frame_congelado is not None and self._bbox_congelado is not None:
                bx = self._bbox_congelado
                img_freeze = self._frame_congelado.copy()
                cv2.rectangle(
                    img_freeze,
                    (int(bx["x1"]), int(bx["y1"])),
                    (int(bx["x2"]), int(bx["y2"])),
                    (0, 255, 0), 2
                )
                try:
                    msg_raw = self._bridge.cv2_to_imgmsg(img_freeze, encoding="bgr8")
                    msg_raw.header.stamp = stamp
                    msg_raw.header.frame_id = "camara_logitech_c270"
                    self._pub.publish(msg_raw)
                except Exception: pass

            # Visor inferior: canvas de carga cyan
            canvas_carga = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                canvas_carga, "ANALIZANDO ELEMENTO...",
                (55, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA
            )
            try:
                msg_seg = self._bridge.cv2_to_imgmsg(canvas_carga, encoding="bgr8")
                msg_seg.header.stamp = stamp
                msg_seg.header.frame_id = "camara_logitech_c270"
                self._pub_video_segmentado.publish(msg_seg)
            except Exception: pass

            # Transicion al estado ESPERA_RETIRO tras 15 frames (~0.5 s)
            if self._frames_analizando >= 15:
                bbox = self._bbox_congelado
                canvas_hsv, estado_limpieza = self._aplicar_segmentacion(
                    self._frame_congelado.copy(), bbox
                )
                self._canvas_veredicto = canvas_hsv

                ancho = int(bbox["x2"]) - int(bbox["x1"])
                alto  = int(bbox["y2"]) - int(bbox["y1"])
                area  = ancho * alto
                tamano_label = "GRANDE" if area > 33000 else "CHICO"
                self._ultimo_veredicto = f"Elemento {tamano_label}: {estado_limpieza}"

                msg_tam = Float32()
                msg_tam.data = float(area * self.k_area)
                self._pub_tamano.publish(msg_tam)

                msg_str = String()
                msg_str.data = self._ultimo_veredicto
                self._pub_analisis.publish(msg_str)

                msg_grados = Float32()
                if estado_limpieza == "OPTIMO":
                    msg_grados.data = 90.0
                    self.get_logger().info("Veredicto OPTIMO → 90.0 grados (aceptacion).")
                else:
                    msg_grados.data = -90.0
                    self.get_logger().info("Veredicto ANOMALIA → -90.0 grados (rechazo).")
                self._pub_comando_grados.publish(msg_grados)

                self._frames_vacio  = 0
                self._estado_actual = 'ESPERA_RETIRO'
                self.get_logger().info(
                    f"Veredicto: {self._ultimo_veredicto}. Transicion a ESPERA_RETIRO."
                )
            return

        # ══════════════════════════════════════════════════════════════════
        # FASE 3: ESPERA_RETIRO — YOLO activo para confirmar retirada
        # ══════════════════════════════════════════════════════════════════
        elif self._estado_actual == 'ESPERA_RETIRO':
            detections = self._inferir(frame)
            mejor_conf = 0.0
            for d in detections:
                if d["conf"] > mejor_conf:
                    mejor_conf = d["conf"]

            # Visor superior: frame congelado permanente
            if self._frame_congelado is not None and self._bbox_congelado is not None:
                bx = self._bbox_congelado
                img_freeze = self._frame_congelado.copy()
                cv2.rectangle(
                    img_freeze,
                    (int(bx["x1"]), int(bx["y1"])),
                    (int(bx["x2"]), int(bx["y2"])),
                    (0, 255, 0), 2
                )
                try:
                    msg_raw = self._bridge.cv2_to_imgmsg(img_freeze, encoding="bgr8")
                    msg_raw.header.stamp = stamp
                    msg_raw.header.frame_id = "camara_logitech_c270"
                    self._pub.publish(msg_raw)
                except Exception: pass

            # Visor inferior: canvas HSV del veredicto
            if self._canvas_veredicto is not None:
                try:
                    msg_seg = self._bridge.cv2_to_imgmsg(self._canvas_veredicto, encoding="bgr8")
                    msg_seg.header.stamp = stamp
                    msg_seg.header.frame_id = "camara_logitech_c270"
                    self._pub_video_segmentado.publish(msg_seg)
                except Exception: pass

            # Publicar veredicto en clasificacion
            msg_str = String()
            if "OPTIMO" in self._ultimo_veredicto:
                msg_str.data = f"{self._ultimo_veredicto}\nEsperando retiro del elemento."
            else:
                msg_str.data = f"{self._ultimo_veredicto}\nPor favor, retire el elemento."

            if mejor_conf >= 0.40:
                self._frames_vacio = 0
            else:
                self._frames_vacio += 1
            self._pub_analisis.publish(msg_str)

            # Transicion de vuelta a BUSQUEDA
            if self._frames_vacio >= 30:
                # Comando home al actuador
                msg_home = Float32()
                msg_home.data = 0.0
                self._pub_comando_grados.publish(msg_home)
                self.get_logger().info("Objeto retirado → 0.0 grados (home). Retorno a BUSQUEDA.")

                # Resetear visor inferior con canvas negro
                frame_negro = np.zeros((480, 640, 3), dtype=np.uint8)
                try:
                    msg_negro = self._bridge.cv2_to_imgmsg(frame_negro, encoding="bgr8")
                    msg_negro.header.stamp = self.get_clock().now().to_msg()
                    msg_negro.header.frame_id = "camara_logitech_c270"
                    self._pub_video_segmentado.publish(msg_negro)
                except Exception: pass

                # Limpiar estado
                self._frames_botella    = 0
                self._frames_analizando = 0
                self._frames_vacio      = 0
                self._frame_congelado   = None
                self._bbox_congelado    = None
                self._canvas_veredicto  = None
                self._estado_actual     = 'BUSQUEDA'

                msg_str = String()
                msg_str.data = "vacio"
                self._pub_analisis.publish(msg_str)
            return

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
