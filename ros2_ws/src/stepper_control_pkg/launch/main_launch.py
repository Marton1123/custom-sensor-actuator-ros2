"""
main_launch.py — Orquestador de producción para stepper_control_pkg.

Lanza los nodos del sistema de clasificacion autonoma en el mismo
proceso de ROS 2 (Jazzy Jalisco):
    1. nodo_actuadores — control GPIO del motor paso a paso, HX711 y sensores IR (lgpio).
    2. nodo_camara     — vision autonoma UVC + inferencia NCNN.
    3. nodo_vision     — clasificacion y segmentacion de objetos.
    4. nodo_gui        — dashboard HMI PyQt6, suscrito a /camara/video_raw.
(Nota: nodo_balanza queda deshabilitado para evitar conflictos sobre los pines GPIO del HX711).

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
            config_path,
            {
                "gpio_chip":     4,       # /dev/gpiochip4 en RPi5
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

    # nodo_balanza deshabilitado: el sensor HX711 es controlado directamente por nodo_actuadores
    # nodo_balanza = Node(
    #     package="stepper_control_pkg",
    #     executable="nodo_balanza",
    #     name="nodo_balanza",
    #     output="screen",
    #     emulate_tty=True,
    #     parameters=[config_path]
    # )

    nodo_vision = Node(
        package="ros2_vision_pkg",
        executable="nodo_vision",
        name="nodo_vision",
        output="screen",
        emulate_tty=True,
        parameters=[
            config_path,
            {
                "modelo_dir": os.path.expanduser('~/custom-sensor-actuator-ros2/IA/models/botellas_vs_latas_ncnn')
            }
        ]
    )

    return LaunchDescription([
        nodo_actuadores,
        nodo_camara,
        nodo_vision,
        nodo_gui,
    ])
