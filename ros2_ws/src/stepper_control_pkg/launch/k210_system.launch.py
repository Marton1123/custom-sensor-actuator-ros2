#!/usr/bin/env python3
"""Sistema completo usando la UnitV K210 como cámara y detector."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("stepper_control_pkg")
    config_path = os.path.join(pkg_share, "config", "parametros.yaml")

    serial_port = LaunchConfiguration("serial_port")
    baudrate = LaunchConfiguration("baudrate")

    return LaunchDescription([
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baudrate", default_value="115200"),
        Node(
            package="ros2_vision_pkg",
            executable="nodo_k210_serial",
            name="nodo_k210_serial",
            output="screen",
            parameters=[{
                "serial_port": serial_port,
                "baudrate": ParameterValue(baudrate, value_type=int),
            }],
        ),
        Node(
            package="ros2_vision_pkg",
            executable="nodo_vision",
            name="nodo_vision",
            output="screen",
            parameters=[{
                "inference_backend": "k210",
                "conf_threshold": 0.45,
            }],
        ),
        Node(
            package="stepper_control_pkg",
            executable="nodo_actuadores",
            name="nodo_actuadores",
            output="screen",
            parameters=[{
                "gpio_chip": 4,
                "pin_step": 17,
                "pin_dir": 27,
                "pasos_por_rev": 3200,
                "delay_pulso": 0.0005,
            }],
        ),
        Node(
            package="stepper_control_pkg",
            executable="nodo_balanza",
            name="nodo_balanza",
            output="screen",
            parameters=[config_path],
        ),
        Node(
            package="stepper_control_pkg",
            executable="nodo_gui",
            name="nodo_gui",
            output="screen",
        ),
    ])
