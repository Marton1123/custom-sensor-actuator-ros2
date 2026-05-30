#!/usr/bin/env python3
"""
nodo_vision.py
Nodo de inferencia NCNN para modelo YOLOv8n (2 clases: bottle vs can).
"""

import sys
import os
import cv2
import numpy as np
import math

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
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

class NodoVision(Node):
    def __init__(self):
        super().__init__('nodo_vision')

        # Declarar parámetros
        self.declare_parameter('modelo_dir', os.path.expanduser('~/ros2_ws/models/botellas_vs_latas_ncnn'))
        self.declare_parameter('conf_threshold', 0.70)
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
        self.get_logger().info(f"Parametros: conf={self.conf_threshold}, iou={self.iou_threshold}, threads={self.num_threads}")

        # Clases según 02_train.py
        self.classes = {0: "bottle", 1: "can"}
        
        # Inicializar NCNN
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # Cambiar a True si se compila NCNN con Vulkan y se requiere GPU
        self.net.opt.num_threads = self.num_threads

        param_path = os.path.join(self.modelo_dir, 'model.ncnn.param')
        bin_path = os.path.join(self.modelo_dir, 'model.ncnn.bin')
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            self.get_logger().error(f"Archivos de modelo no encontrados en {self.modelo_dir}")
            sys.exit(1)

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # Precomputar anclas YOLOv8 (640x640)
        self.anchors, self.strides = generate_anchors(
            strides=[8, 16, 32], 
            grid_sizes=[self.input_size // 8, self.input_size // 16, self.input_size // 32]
        )
        self.dfl_weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1)

        # Configurar QoS BEST_EFFORT
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

        # Publicadores
        self.pub_objeto = self.create_publisher(String, '/objeto_detectado', 10)
        self.pub_tamano = self.create_publisher(Float32, '/tamano_estimado', 10)
        self.pub_bottle = self.create_publisher(Bool, '/botella_detectada', 10)
        self.pub_can = self.create_publisher(Bool, '/lata_detectada', 10)

        self.get_logger().info("Nodo_vision iniciado correctamente. Esperando imágenes...")

    def decode_yolov8(self, raw_output, img_w, img_h, scale_w, scale_h, pad_w, pad_h):
        # raw_output shape esperado: (66, 8400) -> 64 para DFL (regresión) + 2 clases
        if raw_output.shape[0] != 66 or raw_output.shape[1] != 8400:
            self.get_logger().warn(f"Salida de red inesperada: {raw_output.shape}")
            return [], [], []

        coords_raw = raw_output[:64, :]
        cls_raw = raw_output[64:, :]

        # Sigmoide para las clases
        cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))
        
        max_scores = np.max(cls_scores, axis=0)
        class_ids = np.argmax(cls_scores, axis=0)

        valid_mask = max_scores > self.conf_threshold

        if not np.any(valid_mask):
            return [], [], []

        valid_scores = max_scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        
        valid_coords = coords_raw[:, valid_mask].reshape(4, 16, -1)
        valid_anchors = self.anchors[valid_mask].T  # (2, N)
        valid_strides = self.strides[valid_mask].T  # (1, N)

        # DFL Softmax
        x = softmax(valid_coords, axis=1)
        dfl_coords = np.sum(x * self.dfl_weights, axis=1)  # (4, N)

        # Restaurar a xyxy escalado (l, t, r, b)
        x1 = (valid_anchors[0] - dfl_coords[0]) * valid_strides[0]
        y1 = (valid_anchors[1] - dfl_coords[1]) * valid_strides[0]
        x2 = (valid_anchors[0] + dfl_coords[2]) * valid_strides[0]
        y2 = (valid_anchors[1] + dfl_coords[3]) * valid_strides[0]

        # Ajustar pads y escalar a proporciones originales
        x1 = (x1 - pad_w) / scale_w
        y1 = (y1 - pad_h) / scale_h
        x2 = (x2 - pad_w) / scale_w
        y2 = (y2 - pad_h) / scale_h

        # Limitar dentro de la imagen
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # nms separado por clase
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

    def image_callback(self, msg):
        try:
            # ROS a OpenCV (bgr8 para proceso)
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error al procesar imagen: {e}")
            return

        img_h, img_w = img.shape[:2]

        # Preprocesamiento YOLO (Letterbox)
        scale = min(self.input_size / img_w, self.input_size / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_img

        # Inferencia NCNN
        # Convertir a formato NCNN mat (BGR a RGB implícito por YOLO)
        mat_in = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.input_size, self.input_size)
        
        # Normalizar si YOLO exportado lo demanda (Ultralytics usualmente es x/255)
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        ret, mat_out = ex.extract("out0")
        if ret != 0:
            self.get_logger().error("Error en extraccion de red")
            return

        raw_output = np.array(mat_out)
        boxes, scores, class_ids = self.decode_yolov8(raw_output, img_w, img_h, scale, scale, pad_w, pad_h)
        
        # Seleccionar mejor predicción (mayor score)
        best_obj = ""
        best_area = 0.0
        best_score = 0.0

        if len(scores) > 0:
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            cls_num = class_ids[best_idx]
            best_obj = self.classes.get(cls_num, "")

            # Area en cm2 (estimado k_area)
            bx1, by1, bx2, by2 = boxes[best_idx]
            best_area = (bx2 - bx1) * (by2 - by1) * self.k_area

        # Publicar resultados
        msg_obj = String()
        msg_obj.data = best_obj
        self.pub_objeto.publish(msg_obj)

        msg_tamano = Float32()
        msg_tamano.data = float(best_area)
        self.pub_tamano.publish(msg_tamano)

        msg_bottle = Bool()
        msg_bottle.data = (best_obj == "bottle")
        self.pub_bottle.publish(msg_bottle)

        msg_can = Bool()
        msg_can.data = (best_obj == "can")
        self.pub_can.publish(msg_can)

    def destroy_node(self):
        self.get_logger().info("Limpiando recursos NCNN...")
        if self.net:
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
