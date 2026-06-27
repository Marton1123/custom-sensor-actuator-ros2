#!/usr/bin/env python3
"""Carga archivos en MaixPy-v1 usando el REPL normal (sin raw REPL)."""

import argparse
import time
from pathlib import Path

import serial


PROMPT = b">>> "


def recover_repl_baud(port, old_baudrate, new_baudrate):
    """Interrumpe main.py y devuelve el REPL a una velocidad conocida."""
    print(
        "Recuperando REPL de %d a %d bps..." % (old_baudrate, new_baudrate)
    )
    with serial.Serial(port, old_baudrate, timeout=0.2) as connection:
        connection.write(b"\x03\x03")
        connection.flush()
        time.sleep(0.5)
        command = (
            "from machine import UART;"
            "UART.repl_uart().init(%d,8,None,1,read_buf_len=2048)\r\n"
            % new_baudrate
        )
        connection.write(command.encode("ascii"))
        connection.flush()
        time.sleep(0.75)
    print("Comando de recuperación enviado.")


class FriendlyRepl:
    def __init__(self, port, baudrate=115200, wait_for_port=0.0):
        deadline = time.monotonic() + wait_for_port
        while True:
            try:
                self.serial = serial.Serial(port, baudrate, timeout=0.2)
                break
            except serial.SerialException:
                if wait_for_port <= 0 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        time.sleep(0.2)
        self.serial.reset_input_buffer()
        self._interrupt_until_prompt(5.0)

    def close(self):
        self.serial.close()

    def _read_until_prompt(self, timeout):
        deadline = time.monotonic() + timeout
        data = bytearray()
        while time.monotonic() < deadline:
            chunk = self.serial.read(self.serial.in_waiting or 1)
            if chunk:
                data.extend(chunk)
                if PROMPT in data:
                    return bytes(data)
        raise TimeoutError(
            "MaixPy no respondió con >>>. Cierra mpremote/terminales que usen el puerto."
        )

    def _interrupt_until_prompt(self, timeout):
        deadline = time.monotonic() + timeout
        data = bytearray()
        while time.monotonic() < deadline:
            self.serial.write(b"\x03")
            self.serial.flush()
            slice_deadline = time.monotonic() + 0.2
            while time.monotonic() < slice_deadline:
                chunk = self.serial.read(self.serial.in_waiting or 1)
                if chunk:
                    data.extend(chunk)
                    if PROMPT in data:
                        return
                    if len(data) > 8192:
                        del data[:-4096]
        raise TimeoutError(
            "MaixPy no respondió con >>>. Reinicia la UnitV usando --wait-for-port."
        )

    def command(self, source, timeout=5.0):
        if "\n" in source or "\r" in source:
            raise ValueError("Los comandos del REPL deben ocupar una sola línea")
        self.serial.reset_input_buffer()
        self.serial.write(source.encode("utf-8") + b"\r\n")
        self.serial.flush()
        response = self._read_until_prompt(timeout)
        if b"Traceback (most recent call last)" in response:
            raise RuntimeError(response.decode("utf-8", errors="replace"))
        return response

    def upload(self, local_path, remote_path, chunk_size=96):
        data = Path(local_path).read_bytes()
        self.command("import ubinascii")
        self.command("_upload_file=open(%r,'wb')" % remote_path)
        try:
            for offset in range(0, len(data), chunk_size):
                chunk = data[offset:offset + chunk_size].hex()
                self.command(
                    "_upload_file.write(ubinascii.unhexlify(%r))" % chunk
                )
        finally:
            self.command("_upload_file.close()")
        self.command(
            "print('UPLOAD_OK',%r,__import__('os').stat(%r)[6])"
            % (remote_path, remote_path)
        )


def install_camera(repl, script_dir):
    repl.command("import os")
    listing = repl.command("print(os.listdir('/flash'))").decode(
        "utf-8", errors="replace"
    )
    if "boot_m5stickv.py" not in listing:
        repl.command(
            "os.rename('/flash/boot.py','/flash/boot_m5stickv.py')"
        )
        print("Respaldo creado: /flash/boot_m5stickv.py")

    repl.upload(script_dir / "unitv_boot.py", "/flash/boot.py")
    print("Instalado: /flash/boot.py")
    repl.upload(script_dir / "unitv_camera_only.py", "/flash/main.py")
    print("Instalado: /flash/main.py")
    print("Instalación terminada. Desconecta y vuelve a alimentar la UnitV.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Puerto serial, por ejemplo COM5")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--recover-from-baud",
        type=int,
        help="Antes de cargar, recupera el REPL desde esta velocidad",
    )
    parser.add_argument(
        "--wait-for-port",
        type=float,
        default=0.0,
        metavar="SEGUNDOS",
        help="Espera a que aparezca el puerto tras conectar/reiniciar la UnitV",
    )
    parser.add_argument(
        "--install-camera",
        action="store_true",
        help="Respalda el demo M5StickV e instala el firmware de cámara",
    )
    parser.add_argument("--local", type=Path)
    parser.add_argument("--remote")
    args = parser.parse_args()

    if not args.install_camera and not (args.local and args.remote):
        parser.error("usa --install-camera o especifica --local y --remote")

    if args.recover_from_baud:
        recover_repl_baud(args.port, args.recover_from_baud, args.baudrate)

    repl = FriendlyRepl(args.port, args.baudrate, args.wait_for_port)
    try:
        if args.install_camera:
            install_camera(repl, Path(__file__).resolve().parent)
        else:
            repl.upload(args.local.resolve(), args.remote)
    finally:
        repl.close()


if __name__ == "__main__":
    main()
