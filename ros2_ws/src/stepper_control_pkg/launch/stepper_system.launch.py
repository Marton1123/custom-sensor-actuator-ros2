#!/usr/bin/env python3
"""Launch principal para sensores + actuadores + visión (YOLO NCNN)."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    # ── Argumentos declarados (sobreescribibles desde CLI) ────────────────
    args = [
        DeclareLaunchArgument("sample_rate", default_value="10"),
        DeclareLaunchArgument("gpio_chip", default_value="4"),
        DeclareLaunchArgument("pin_step", default_value="17"),
        DeclareLaunchArgument("pin_dir", default_value="27"),
        DeclareLaunchArgument("pasos_por_rev", default_value="3200"),
        DeclareLaunchArgument("delay_pulso", default_value="0.0005"),
        DeclareLaunchArgument("modelo_ncnn", default_value="/home/pi/modelos/best_ncnn_model"),
        DeclareLaunchArgument("cam_id", default_value="0"),
        DeclareLaunchArgument("cam_width", default_value="640"),
        DeclareLaunchArgument("cam_height", default_value="480"),
        DeclareLaunchArgument("cam_fps", default_value="30"),
        DeclareLaunchArgument("infer_hz", default_value="10.0"),
        DeclareLaunchArgument("k_area", default_value="0.05"),
        DeclareLaunchArgument("conf_threshold", default_value="0.70"),
    ]

    nodo_sensores = Node(
        package="stepper_control_pkg",
        executable="nodo_sensores",
        name="nodo_sensores",
        output="screen",
        parameters=[{
            "sample_rate": LaunchConfiguration("sample_rate"),
        }],
    )

    nodo_actuadores = Node(
        package="stepper_control_pkg",
        executable="nodo_actuadores",
        name="nodo_actuadores",
        output="screen",
        parameters=[{
            "gpio_chip": LaunchConfiguration("gpio_chip"),
            "pin_step": LaunchConfiguration("pin_step"),
            "pin_dir": LaunchConfiguration("pin_dir"),
            "pasos_por_rev": LaunchConfiguration("pasos_por_rev"),
            "delay_pulso": LaunchConfiguration("delay_pulso"),
        }],
    )

    nodo_vision = Node(
        package="ros2_vision_pkg",
        executable="nodo_vision",
        name="nodo_vision",
        output="screen",
        parameters=[{
            "modelo_ncnn": LaunchConfiguration("modelo_ncnn"),
            "cam_id": LaunchConfiguration("cam_id"),
            "cam_width": LaunchConfiguration("cam_width"),
            "cam_height": LaunchConfiguration("cam_height"),
            "cam_fps": LaunchConfiguration("cam_fps"),
            "infer_hz": LaunchConfiguration("infer_hz"),
            "k_area": LaunchConfiguration("k_area"),
            "conf_threshold": LaunchConfiguration("conf_threshold"),
        }],
    )

    return LaunchDescription(args + [nodo_sensores, nodo_actuadores, nodo_vision])
