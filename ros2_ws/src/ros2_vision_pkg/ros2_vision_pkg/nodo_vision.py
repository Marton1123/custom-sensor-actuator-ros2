#!/usr/bin/env python3
"""
nodo_vision.py
Nodo de inferencia NCNN para modelo YOLOv8n (2 clases: bottle vs can).
Restaurado con Máquina de Estados (FSM) y SPoP.
"""

import sys
import os
import cv2
import numpy as np
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool
from cv_bridge import CvBridge

import ncnn

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def generate_anchors(strides=[8, 16, 32], grid_sizes=[80, 40, 20]):
    anchor_points = []
    stride_tensor = []
    for g, s in zip(grid_sizes, strides):
        shift_x = np.arange(g) + 0.5
        shift_y = np.arange(g) + 0.5
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        anchor_points.append(np.stack([shift_x.flatten(), shift_y.flatten()], axis=1))
        stride_tensor.append(np.full((g * g, 1), s))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

class NodoVision(Node):
    def __init__(self):
        super().__init__('nodo_vision')

        # Declarar parámetros
        ruta_modelo = os.path.expanduser('~/custom-sensor-actuator-ros2/IA/models/botellas_vs_latas_ncnn')
        self.declare_parameter('modelo_dir', ruta_modelo)
        self.declare_parameter('conf_threshold', 0.45) # Bajado a 0.45 para latas
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('k_area', 0.05)
        self.declare_parameter('num_threads', 4)

        # Leer parámetros
        self.modelo_dir = self.get_parameter('modelo_dir').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.k_area = self.get_parameter('k_area').get_parameter_value().double_value
        self.num_threads = self.get_parameter('num_threads').get_parameter_value().integer_value

        self.get_logger().info(f"Cargando modelo NCNN desde: {self.modelo_dir}")
        
        self.classes = {0: "bottle", 1: "can"}
        
        # FSM Inicializacion
        self._estado_actual = 'BUSQUEDA'
        self._frames_botella = 0
        self._frames_vacio = 0
        self._frames_analizando = 0
        self._frame_congelado = None
        self._id_congelado = None
        self._box_congelado = None
        self._ultimo_veredicto = "vacio"
        self._lost_frames = 0
        self._last_best_obj = None
        self._last_best_box = None
        self._last_best_score = 0.0
        self._ultimo_estado_publicado = ""
        
        # Hilo de inferencia
        self._inference_lock = threading.Lock()
        self._frame_for_inference = None
        self._latest_inference = {"best_obj": "", "best_score": 0.0, "best_box": None}
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        
        # Pre-allocate canvas
        self._canvas_espera = np.zeros((480, 640, 3), dtype=np.uint8)
        self._canvas_analizando = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # NCNN init
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = self.num_threads

        param_path = os.path.join(self.modelo_dir, 'model.ncnn.param')
        bin_path = os.path.join(self.modelo_dir, 'model.ncnn.bin')
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            self.get_logger().error(f"Archivos de modelo no encontrados en {self.modelo_dir}")
            sys.exit(1)

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self.anchors, self.strides = generate_anchors(
            strides=[8, 16, 32], 
            grid_sizes=[self.input_size // 8, self.input_size // 16, self.input_size // 32]
        )
        self.dfl_weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.sub_camara = self.create_subscription(
            Image,
            '/camara/video_raw',
            self.image_callback,
            qos_profile
        )

        # Publicadores unificados (SPoP)
        self.pub_raw = self.create_publisher(Image, '/camara/video_procesado', qos_profile)
        self.pub_seg = self.create_publisher(Image, '/camara/video_segmentado', qos_profile)
        self.pub_clasificacion = self.create_publisher(String, '/clasificacion_objeto', 10)
        self.pub_tamano = self.create_publisher(Float32, '/tamano_estimado', 10)
        self.pub_comando_grados = self.create_publisher(Float32, '/comando_grados', 10)

        self.get_logger().info("Nodo_vision iniciado (FSM + SPoP). Esperando imagenes...")

    def _aplicar_segmentacion(self, frame_bgr, bbox):
        frame_copia = frame_bgr.copy()
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_copia.shape[1], x2), min(frame_copia.shape[0], y2)

        if (x2 <= x1) or (y2 <= y1):
            return frame_copia, "OPTIMO"

        roi = frame_copia[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_dark = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([179, 255, 80]))
        mask_sat = cv2.inRange(roi_hsv, np.array([0, 100, 0]), np.array([179, 255, 255]))
        mask_anomalia = cv2.bitwise_or(mask_dark, mask_sat)

        total_pixels = roi.shape[0] * roi.shape[1]
        anomalia_pixels = np.count_nonzero(mask_anomalia)
        porcentaje = anomalia_pixels / total_pixels if total_pixels > 0 else 0.0

        estado_limpieza = "OPTIMO" if porcentaje < 0.10 else "ANOMALIA"

        anomalia_bgr = np.zeros_like(roi)
        anomalia_bgr[mask_anomalia > 0] = [0, 0, 255]

        canvas = cv2.addWeighted(frame_copia, 0.3, np.zeros_like(frame_copia), 0, 0)
        canvas[y1:y2, x1:x2] = anomalia_bgr

        return canvas, estado_limpieza

    def decode_yolov8(self, raw_output, img_w, img_h, scale_w, scale_h, pad_w, pad_h):
        num_classes = len(self.classes)
        expected_end2end = 4 + num_classes  # 6 para 2 clases

        if raw_output.shape[0] == expected_end2end and raw_output.shape[1] == 8400:
            # ── Modelo end2end: boxes ya decodificadas + scores con sigmoid ──
            cx = raw_output[0, :]
            cy = raw_output[1, :]
            w  = raw_output[2, :]
            h  = raw_output[3, :]
            cls_scores = raw_output[4:, :]  # [num_classes, 8400], ya sigmoid

            max_scores = np.max(cls_scores, axis=0)
            class_ids  = np.argmax(cls_scores, axis=0)

            valid_mask = max_scores > self.conf_threshold
            if not np.any(valid_mask):
                return [], [], []

            v_cx = cx[valid_mask]
            v_cy = cy[valid_mask]
            v_w  = w[valid_mask]
            v_h  = h[valid_mask]
            valid_scores    = max_scores[valid_mask]
            valid_class_ids = class_ids[valid_mask]

            # cx,cy,w,h → x1,y1,x2,y2
            x1 = v_cx - v_w / 2.0
            y1 = v_cy - v_h / 2.0
            x2 = v_cx + v_w / 2.0
            y2 = v_cy + v_h / 2.0

            # Quitar padding y reescalar a imagen original
            x1 = (x1 - pad_w) / scale_w
            y1 = (y1 - pad_h) / scale_h
            x2 = (x2 - pad_w) / scale_w
            y2 = (y2 - pad_h) / scale_h

            x1 = np.clip(x1, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            x2 = np.clip(x2, 0, img_w)
            y2 = np.clip(y2, 0, img_h)

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            final_boxes, final_scores, final_class_ids = [], [], []
            for cls_id in np.unique(valid_class_ids):
                cls_mask = valid_class_ids == cls_id
                cls_boxes = boxes[cls_mask]
                cls_scores_arr = valid_scores[cls_mask]
                keep = nms(cls_boxes, cls_scores_arr, self.iou_threshold)
                for k in keep:
                    final_boxes.append(cls_boxes[k])
                    final_scores.append(cls_scores_arr[k])
                    final_class_ids.append(cls_id)

            return final_boxes, final_scores, final_class_ids

        elif raw_output.shape[0] == (64 + num_classes) and raw_output.shape[1] == 8400:
            # ── Modelo RAW: DFL sin procesar + logits sin sigmoid ──
            coords_raw = raw_output[:64, :]
            cls_raw = raw_output[64:, :]

            cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))
            max_scores = np.max(cls_scores, axis=0)
            class_ids = np.argmax(cls_scores, axis=0)

            valid_mask = max_scores > self.conf_threshold
            if not np.any(valid_mask):
                return [], [], []

            valid_scores = max_scores[valid_mask]
            valid_class_ids = class_ids[valid_mask]

            valid_coords = coords_raw[:, valid_mask].reshape(4, 16, -1)
            valid_anchors = self.anchors[valid_mask].T
            valid_strides = self.strides[valid_mask].T

            x = softmax(valid_coords, axis=1)
            dfl_coords = np.sum(x * self.dfl_weights, axis=1)

            x1 = (valid_anchors[0] - dfl_coords[0]) * valid_strides[0]
            y1 = (valid_anchors[1] - dfl_coords[1]) * valid_strides[0]
            x2 = (valid_anchors[0] + dfl_coords[2]) * valid_strides[0]
            y2 = (valid_anchors[1] + dfl_coords[3]) * valid_strides[0]

            x1 = (x1 - pad_w) / scale_w
            y1 = (y1 - pad_h) / scale_h
            x2 = (x2 - pad_w) / scale_w
            y2 = (y2 - pad_h) / scale_h

            x1 = np.clip(x1, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            x2 = np.clip(x2, 0, img_w)
            y2 = np.clip(y2, 0, img_h)

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            final_boxes, final_scores, final_class_ids = [], [], []
            for cls_id in np.unique(valid_class_ids):
                cls_mask = valid_class_ids == cls_id
                cls_boxes = boxes[cls_mask]
                cls_scores_arr = valid_scores[cls_mask]
                keep = nms(cls_boxes, cls_scores_arr, self.iou_threshold)
                for k in keep:
                    final_boxes.append(cls_boxes[k])
                    final_scores.append(cls_scores_arr[k])
                    final_class_ids.append(cls_id)

            return final_boxes, final_scores, final_class_ids

        else:
            self.get_logger().warn(
                f"Formato de salida inesperado: {raw_output.shape}. "
                f"Esperado ({expected_end2end}, 8400) o ({64 + num_classes}, 8400)."
            )
            return [], [], []

    def _inference_loop(self):
        while rclpy.ok():
            with self._inference_lock:
                frame = self._frame_for_inference
                self._frame_for_inference = None

            if frame is None:
                time.sleep(0.01)
                continue

            img_h, img_w = frame.shape[:2]
            scale = min(self.input_size / img_w, self.input_size / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pad_w = (self.input_size - new_w) // 2
            pad_h = (self.input_size - new_h) // 2

            resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
            canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_img

            mat_in = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.input_size, self.input_size)
            mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            ex = self.net.create_extractor()
            ex.input("in0", mat_in)
            ret, mat_out = ex.extract("out0")

            best_obj = ""
            best_score = 0.0
            best_box = None

            if ret == 0:
                raw_output = np.array(mat_out)
                boxes, scores, class_ids = self.decode_yolov8(raw_output, img_w, img_h, scale, scale, pad_w, pad_h)
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    cls_num = class_ids[best_idx]
                    best_obj = self.classes.get(cls_num, "desconocido")
                    best_box = boxes[best_idx]
            
            with self._inference_lock:
                self._latest_inference = {"best_obj": best_obj, "best_score": best_score, "best_box": best_box}

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        img_h, img_w = img.shape[:2]
        out_raw = img.copy()
        out_seg = self._canvas_espera.copy()
        out_estado = self._ultimo_veredicto
        out_area = 0.0

        scale = min(self.input_size / img_w, self.input_size / img_h)
        
        with self._inference_lock:
            if self._estado_actual in ['BUSQUEDA', 'ESPERA_RETIRO'] and self._frame_for_inference is None:
                self._frame_for_inference = img.copy()
            best_obj = self._latest_inference["best_obj"]
            best_score = self._latest_inference["best_score"]
            best_box = self._latest_inference["best_box"]

        # ---------------- FSM ----------------
        if self._estado_actual == 'BUSQUEDA':
            out_estado = "vacio"
            
            if best_obj in ["bottle", "can"]:
                self._frames_botella += 1
                self._lost_frames = 0
                self._last_best_obj = best_obj
                self._last_best_box = best_box
                self._last_best_score = best_score
            elif hasattr(self, '_lost_frames') and self._lost_frames < 5 and self._last_best_obj is not None:
                self._lost_frames += 1
                best_obj = self._last_best_obj
                best_box = self._last_best_box
                best_score = self._last_best_score
            else:
                self._frames_botella = 0
                self._last_best_obj = None
            
            if best_obj in ["bottle", "can"]:
                bx1, by1, bx2, by2 = map(int, best_box)
                cv2.rectangle(out_raw, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(out_raw, f"{best_obj.upper()} {best_score:.2f}", (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if self._frames_botella >= 15:
                self._frame_congelado = img.copy()
                self._id_congelado = best_obj
                self._box_congelado = best_box
                self._estado_actual = 'ANALIZANDO'
                self._frames_analizando = 0
                self._ultimo_veredicto = "analizando"
                out_estado = "analizando"

        elif self._estado_actual == 'ANALIZANDO':
            self._frames_analizando += 1
            bx1, by1, bx2, by2 = map(int, self._box_congelado)
            
            clase_str = "LATA" if self._id_congelado == "can" else "BOTELLA"
            cv2.putText(out_raw, f"ANALIZANDO {clase_str}...", (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.rectangle(out_raw, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            
            out_seg = self._canvas_analizando.copy()
            out_estado = "analizando"

            if self._frames_analizando >= 15:
                self._estado_actual = 'ESPERA_RETIRO'
                
                area = (bx2 - bx1) * (by2 - by1) * self.k_area
                msg_grados = Float32()

                if self._id_congelado == "bottle":
                    seg_img, estado_limpieza = self._aplicar_segmentacion(self._frame_congelado, self._box_congelado)
                    if estado_limpieza == "OPTIMO":
                        self._ultimo_veredicto = "¡Botella Aceptada!"
                        msg_grados.data = 90.0
                    else:
                        self._ultimo_veredicto = "Botella Rechazada (Por favor, enjuáguela)"
                        msg_grados.data = 0.0
                else:
                    self._ultimo_veredicto = "¡Lata Aceptada!"
                    msg_grados.data = -90.0
                
                out_estado = self._ultimo_veredicto
                
                self.pub_comando_grados.publish(msg_grados)
                self._frames_vacio = 0

        elif self._estado_actual == 'ESPERA_RETIRO':
            if best_obj in ["bottle", "can"] and best_score > 0.40:
                self._frames_vacio = 0
            else:
                self._frames_vacio += 1

            if self._id_congelado == "bottle":
                if "Aceptada" in self._ultimo_veredicto:
                    out_seg = np.zeros((480, 640, 3), dtype=np.uint8)
                    out_seg[:] = (0, 100, 0) # Verde
                else:
                    seg_img, _ = self._aplicar_segmentacion(self._frame_congelado, self._box_congelado)
                    out_seg = seg_img
            else:
                out_seg = np.zeros((480, 640, 3), dtype=np.uint8)
                out_seg[:] = (150, 0, 0) # Azul oscuro
            
            bx1, by1, bx2, by2 = map(int, self._box_congelado)
            cv2.rectangle(out_raw, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            clase_str = "LATA" if self._id_congelado == "can" else "BOTELLA"
            cv2.putText(out_raw, f"RETIRE {clase_str}", (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            out_estado = self._ultimo_veredicto

            if self._frames_vacio >= 10:
                self._estado_actual = 'RESETEO'

        elif self._estado_actual == 'RESETEO':
            self._estado_actual = 'BUSQUEDA'
            self._frames_botella = 0
            self._last_best_obj = None
            out_estado = "vacio"
            msg_grados = Float32()
            msg_grados.data = 0.0
            self.pub_comando_grados.publish(msg_grados)

        # ----------------------------------------------------
        # SPoP - Single Point of Publication
        # ----------------------------------------------------
        out_raw_rgb = cv2.cvtColor(out_raw, cv2.COLOR_BGR2RGB)
        out_seg_rgb = cv2.cvtColor(out_seg, cv2.COLOR_BGR2RGB)

        msg_raw = self.bridge.cv2_to_imgmsg(out_raw_rgb, encoding="rgb8")
        msg_raw.header.stamp = msg.header.stamp
        msg_raw.header.frame_id = "nodo_vision_procesado_raw"
        
        msg_seg = self.bridge.cv2_to_imgmsg(out_seg_rgb, encoding="rgb8")
        msg_seg.header.stamp = msg.header.stamp
        msg_seg.header.frame_id = "nodo_vision_procesado_seg"

        self.pub_raw.publish(msg_raw)
        self.pub_seg.publish(msg_seg)

        if out_estado != self._ultimo_estado_publicado:
            msg_obj = String()
            msg_obj.data = out_estado
            self.pub_clasificacion.publish(msg_obj)
            self._ultimo_estado_publicado = out_estado
        
        msg_tam = Float32()
        msg_tam.data = out_area
        self.pub_tamano.publish(msg_tam)

    def destroy_node(self):
        if hasattr(self, 'net') and self.net:
            self.net.clear()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    nodo = NodoVision()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
