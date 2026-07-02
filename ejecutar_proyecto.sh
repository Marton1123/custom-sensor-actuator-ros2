#!/bin/bash
# =============================================================================
# ejecutar_proyecto.sh
# =============================================================================
# Script para iniciar el proyecto en segundo plano y limpiar todo al cerrar.
# Optimizado para el usuario final (arranque instantáneo sin compilación).
# =============================================================================

# 1. Limpieza inicial de procesos colgados
pkill -9 -f ros2
pkill -9 -f stepper_control
pkill -9 -f nodo_
fuser -k /dev/video0 2>/dev/null

# 2. Cargar entorno de ROS 2 y configurar pantalla
source /opt/ros/jazzy/setup.bash
source /home/lab-ros/custom-sensor-actuator-ros2/ros2_ws/install/setup.bash
export DISPLAY=:0

# Asegurar la rotación vertical (portrait) de la pantalla
xrandr --output HDMI-2 --rotate left

# 3. Lanzar la aplicación en primer plano de este script (bloqueante)
ros2 launch stepper_control_pkg main_launch.py

# 4. Cuando el usuario cierre la app (Alt+F4), el script continúa aquí y limpia todo
pkill -9 -f ros2
pkill -9 -f stepper_control
pkill -9 -f nodo_
fuser -k /dev/video0 2>/dev/null
