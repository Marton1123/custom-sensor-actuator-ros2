"""
main_launch.py — Orquestador de producción para stepper_control_pkg.

Lanza los cinco nodos del sistema de clasificación autónoma con ROS 2
(Jazzy Jalisco):
    1. nodo_actuadores — control GPIO del motor paso a paso (lgpio).
    2. nodo_k210_serial — recibe por USB el video de la cámara OV7740.
    3. nodo_vision      — inferencia YOLO/NCNN ejecutada en la Raspberry Pi.
    4. nodo_gui         — dashboard HMI PyQt6.
    5. nodo_balanza     — lectura de celda de carga HX711.

Uso:
    ros2 launch stepper_control_pkg main_launch.py

Parámetros sobreescribibles en tiempo de lanzamiento:
    ros2 launch stepper_control_pkg main_launch.py \
        serial_port:=/dev/ttyUSB0 baudrate:=115200
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    serial_port = LaunchConfiguration("serial_port")
    baudrate = LaunchConfiguration("baudrate")

    nodo_k210_serial = Node(
        package="ros2_vision_pkg",
        executable="nodo_k210_serial",
        name="nodo_k210_serial",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "serial_port": serial_port,
            "baudrate": ParameterValue(baudrate, value_type=int),
        }],
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
            "modelo_dir": os.path.expanduser(
                "~/custom-sensor-actuator-ros2/IA/models/botellas_vs_latas_ncnn"
            ),
            # La K210 sólo entrega imágenes; la inferencia permanece en la Pi.
            "inference_backend": "ncnn",
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyUSB0",
            description="Puerto USB serie de la UnitV K210",
        ),
        DeclareLaunchArgument(
            "baudrate",
            default_value="115200",
            description="Velocidad del flujo JPEG de la UnitV",
        ),
        nodo_actuadores,
        nodo_k210_serial,
        nodo_vision,
        nodo_gui,
        nodo_balanza,
    ])
