#!/usr/bin/env python3
"""
Launch file para arrancar ambos nodos simultáneamente.

Uso (desde el workspace compilado):
    ros2 launch stepper_control_pkg stepper_system.launch.py

Parámetros configurables:
    sample_rate  : frecuencia de muestreo del sensor en Hz     [default: 10]
    sensor_pin   : canal/pin del sensor                        [default: 0]
    umbral       : umbral para activar el motor                [default: 50.0]
    step_delay   : retardo entre pasos del motor (segundos)    [default: 0.002]
    pin_step     : pin STEP del TB6600                         [default: 17]
    pin_dir      : pin DIR del TB6600                          [default: 27]
    pin_ena      : pin ENA del TB6600                          [default: 22]
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    # ── Argumentos declarados (sobreescribibles desde CLI) ────────────────
    args = [
        DeclareLaunchArgument("sample_rate",  default_value="10"),
        DeclareLaunchArgument("sensor_pin",   default_value="0"),
        DeclareLaunchArgument("umbral",       default_value="50.0"),
        DeclareLaunchArgument("step_delay",   default_value="0.002"),
        DeclareLaunchArgument("pin_step",     default_value="17"),
        DeclareLaunchArgument("pin_dir",      default_value="27"),
        DeclareLaunchArgument("pin_ena",      default_value="22"),
    ]

    nodo_sensores = Node(
        package="stepper_control_pkg",
        executable="nodo_sensores",
        name="nodo_sensores",
        output="screen",
        parameters=[{
            "sample_rate": LaunchConfiguration("sample_rate"),
            "sensor_pin":  LaunchConfiguration("sensor_pin"),
        }],
    )

    nodo_actuadores = Node(
        package="stepper_control_pkg",
        executable="nodo_actuadores",
        name="nodo_actuadores",
        output="screen",
        parameters=[{
            "umbral":      LaunchConfiguration("umbral"),
            "step_delay":  LaunchConfiguration("step_delay"),
            "pin_step":    LaunchConfiguration("pin_step"),
            "pin_dir":     LaunchConfiguration("pin_dir"),
            "pin_ena":     LaunchConfiguration("pin_ena"),
        }],
    )

    return LaunchDescription(args + [nodo_sensores, nodo_actuadores])
