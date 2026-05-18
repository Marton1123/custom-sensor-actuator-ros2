"""
kiosco_launch.py
================
Launch file de producción (Modo Kiosco) para stepper_control_pkg.

Lanza dos nodos en el mismo proceso de ROS 2:
    1. nodo_actuadores  — control GPIO del motor NEMA 17 (lgpio)
    2. nodo_gui         — interfaz gráfica táctil PyQt6 + cámara OpenCV

Uso:
    ros2 launch stepper_control_pkg kiosco_launch.py

Parámetros sobreescribibles desde línea de comandos:
    ros2 launch stepper_control_pkg kiosco_launch.py gpio_chip:=4 delay_pulso:=0.001
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Retorna el LaunchDescription con los dos nodos del sistema."""

    nodo_actuadores = Node(
        package="stepper_control_pkg",
        executable="nodo_actuadores",
        name="nodo_actuadores",
        output="screen",
        emulate_tty=True,           # colores y logs legibles en terminal
        parameters=[
            {
                "gpio_chip":      4,        # /dev/gpiochip4 en RPi5
                "pin_step":       17,       # PUL+ → GPIO 17
                "pin_dir":        27,       # DIR+ → GPIO 27
                "pasos_por_rev":  3200,     # 1/16 microstepping en TB6600
                "delay_pulso":    0.0005,   # 1 ms/ciclo ≈ 18.75 RPM
            }
        ],
    )

    nodo_gui = Node(
        package="stepper_control_pkg",
        executable="nodo_gui",
        name="nodo_gui",
        output="screen",
        emulate_tty=True,
        # La GUI necesita acceso al display — DISPLAY se hereda del entorno
        # Si se lanza desde autostart, DISPLAY=:0 ya está disponible.
    )

    return LaunchDescription([
        nodo_actuadores,
        nodo_gui,
    ])
