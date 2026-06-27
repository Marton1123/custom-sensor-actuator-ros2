#!/usr/bin/env python3
"""Valida en PC el flujo JPEG+JSON enviado por la UnitV vía USB-serie."""

import argparse
import json
import struct
import time

import serial


MAGIC = b"K2V1"
HEADER = struct.Struct("<4sIII")
MAX_METADATA = 16 * 1024
MAX_JPEG = 512 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Puerto FTDI, por ejemplo COM5")
    parser.add_argument("--baudrate", type=int, default=115200)
    args = parser.parse_args()

    buffer = bytearray()
    frames = 0
    started = time.monotonic()

    print(f"Escuchando {args.port} a {args.baudrate} bps. Ctrl+C para salir.")
    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.25) as port:
            while True:
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
                        print("Paquete descartado: checksum incorrecto")
                        continue

                    metadata = json.loads(payload[:metadata_len].decode("utf-8"))
                    frames += 1
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        "frame=%s size=%sx%s jpeg=%s bytes fps=%.2f detections=%s"
                        % (
                            metadata.get("seq"),
                            metadata.get("width"),
                            metadata.get("height"),
                            jpeg_len,
                            frames / elapsed,
                            len(metadata.get("detections", [])),
                        )
                    )
    except KeyboardInterrupt:
        print("\nPrueba finalizada.")


if __name__ == "__main__":
    main()
