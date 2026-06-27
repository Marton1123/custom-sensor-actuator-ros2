"""
Firmware MaixPy-v1 para M5Stack UnitV K210.

Ejecuta YOLOv2 en la KPU y transmite cada fotograma JPEG junto con sus
detecciones a la Raspberry Pi. Copiar como /sd/main.py y colocar el modelo
en /sd/bottle_can.kmodel.
"""

import sensor
import time
import ujson
import ustruct
import KPU as kpu
from machine import UART


MODEL_PATH = "/sd/bottle_can.kmodel"
LABELS = ("bottle", "can")

# Pegar aquí la tupla de anchors entregada por maix_train.
ANCHORS = None

BAUDRATE = 115200
JPEG_QUALITY = 20
SEND_INTERVAL_MS = 200
CONFIDENCE = 0.45
NMS = 0.30

MAGIC = b"K2V1"


def init_uart():
    return UART.repl_uart()


def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    # maix_train genera por defecto modelos YOLOv2 de 224x224.
    sensor.set_windowing((224, 224))
    sensor.set_vflip(0)
    sensor.set_hmirror(0)
    sensor.run(1)
    sensor.skip_frames(time=1500)


def encode_detections(objects):
    detections = []
    for obj in objects or ():
        class_id = obj.classid()
        if 0 <= class_id < len(LABELS):
            detections.append({
                "class": LABELS[class_id],
                "class_id": class_id,
                "confidence": obj.value(),
                "x": obj.x(),
                "y": obj.y(),
                "w": obj.w(),
                "h": obj.h(),
            })
    return detections


def send_packet(uart, sequence, img, detections):
    metadata = ujson.dumps({
        "seq": sequence,
        "width": img.width(),
        "height": img.height(),
        "detections": detections,
    }).encode()
    compressed = img.compress(quality=JPEG_QUALITY)
    jpeg = bytearray(compressed)
    payload = bytearray(metadata)
    payload.extend(jpeg)
    checksum = sum(payload) & 0xFFFFFFFF
    header = ustruct.pack("<4sIII", MAGIC, len(metadata), len(jpeg), checksum)
    uart.write(header)
    uart.write(payload)


def main():
    if not ANCHORS:
        raise ValueError("Configura ANCHORS con los valores generados por maix_train")
    uart = init_uart()
    init_camera()
    task = kpu.load(MODEL_PATH)
    kpu.init_yolo2(task, CONFIDENCE, NMS, len(ANCHORS) // 2, ANCHORS)

    sequence = 0
    last_send = time.ticks_ms() - SEND_INTERVAL_MS
    try:
        while True:
            img = sensor.snapshot()
            objects = kpu.run_yolo2(task, img)
            now = time.ticks_ms()
            if time.ticks_diff(now, last_send) >= SEND_INTERVAL_MS:
                send_packet(uart, sequence, img, encode_detections(objects))
                sequence = (sequence + 1) & 0xFFFFFFFF
                last_send = now
    finally:
        kpu.deinit(task)


main()
