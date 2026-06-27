"""
Firmware MaixPy-v1 para usar la M5Stack UnitV como cámara de ROS 2.

No carga modelos ni ejecuta inferencia. Captura una ventana de 224x224,
comprime cada fotograma como JPEG y lo transmite por el USB-serie de la
UnitV usando el mismo protocolo que unitv_bottle_can.py.
"""

import sensor
import time
import ujson
import ustruct
from machine import UART


BAUDRATE = 115200
JPEG_QUALITY = 20
SEND_INTERVAL_MS = 200

MAGIC = b"K2V1"


def init_uart():
    return UART.repl_uart()


def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((224, 224))
    sensor.set_vflip(0)
    sensor.set_hmirror(0)
    sensor.run(1)
    sensor.skip_frames(time=1500)


def send_frame(uart, sequence, img):
    metadata = ujson.dumps({
        "seq": sequence,
        "width": img.width(),
        "height": img.height(),
        "detections": [],
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
    uart = init_uart()
    init_camera()
    sequence = 0
    last_send = time.ticks_ms() - SEND_INTERVAL_MS

    while True:
        img = sensor.snapshot()
        now = time.ticks_ms()
        if time.ticks_diff(now, last_send) >= SEND_INTERVAL_MS:
            send_frame(uart, sequence, img)
            sequence = (sequence + 1) & 0xFFFFFFFF
            last_send = now


main()
