"""
main_launch.py — Orquestador de producción para stepper_control_pkg.

Lanza los cuatro nodos del sistema de clasificación autónoma en el mismo
proceso de ROS 2 (Jazzy Jalisco):
    1. nodo_actuadores — control GPIO del motor paso a paso (lgpio).
    2. nodo_camara     — visión autónoma UVC + inferencia NCNN.
    3. nodo_gui        — dashboard HMI PyQt6, suscrito a /camara/video_raw.
    4. nodo_balanza    — lectura de celda de carga HX711, publica /peso_elemento.

Uso:
    ros2 launch stepper_control_pkg main_launch.py

Parámetros sobreescribibles en tiempo de lanzamiento:
    ros2 launch stepper_control_pkg main_launch.py delay_pulso:=0.001
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Construye el grafo de nodos de ROS 2 para el sistema de clasificación autónoma.

    Carga el archivo de parámetros YAML compartido y registra los cuatro
    nodos del sistema con sus configuraciones de hardware por defecto.

    Returns:
        LaunchDescription: Descripción completa del grafo de lanzamiento.
    """
    pkg_share = get_package_share_directory('stepper_control_pkg')
    config_path = os.path.join(pkg_share, 'config', 'parametros.yaml')

    nodo_actuadores = Node(
        package="stepper_control_pkg",
        executable="nodo_actuadores",
        name="nodo_actuadores",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "gpio_chip":     4,       # /dev/gpiochip4 en RPi5
                "pin_step":      17,      # PUL+ -> GPIO 17
                "pin_dir":       27,      # DIR+ -> GPIO 27
                "pasos_por_rev": 3200,    # 1/16 microstepping en TB6600
                "delay_pulso":   0.0005,  # ~18.75 RPM
            }
        ],
    )

    nodo_camara = Node(
        package="stepper_control_pkg",
        executable="nodo_camara",
        name="nodo_camara",
        output="screen",
        emulate_tty=True,
        parameters=[config_path]
        # /dev/video0 se hereda del entorno; el usuario debe pertenecer al grupo 'video'
    )

    nodo_gui = Node(
        package="stepper_control_pkg",
        executable="nodo_gui",
        name="nodo_gui",
        output="screen",
        emulate_tty=True,
        # DISPLAY=:0 se hereda del entorno de autostart GNOME
    )

    nodo_balanza = Node(
        package="stepper_control_pkg",
        executable="nodo_balanza",
        name="nodo_balanza",
        output="screen",
        emulate_tty=True,
        parameters=[config_path]
    )

    nodo_vision = Node(
        package="ros2_vision_pkg",
        executable="nodo_vision",
        name="nodo_vision",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "modelo_dir": os.path.expanduser('~/ros2_ws/models/botellas_vs_latas_ncnn')
        }]
    )

    return LaunchDescription([
        nodo_actuadores,
        nodo_camara,
        nodo_vision,
        nodo_gui,
        nodo_balanza,
    ])
