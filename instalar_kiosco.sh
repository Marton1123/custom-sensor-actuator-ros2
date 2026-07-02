#!/bin/bash
# =============================================================================
# instalar_kiosco.sh
# =============================================================================
# Script de despliegue automático para el Panel de Control NEMA 17.
# Configura la Raspberry Pi 5 (Ubuntu 24.04) como dispositivo empotrado
# en Modo Kiosco arrancando el sistema ROS 2 al iniciar sesión en GNOME.
#
# Uso (una sola vez tras clonar el repositorio):
#   chmod +x instalar_kiosco.sh
#   ./instalar_kiosco.sh
#
# Requisitos previos:
#   - ROS 2 Jazzy instalado en /opt/ros/jazzy/
#   - Workspace compilado: cd ros2_ws && colcon build
#   - PyQt6 instalado:     pip3 install PyQt6
#   - OpenCV instalado:    sudo apt install python3-opencv
#   - lgpio instalado:     sudo apt install python3-lgpio
# =============================================================================

set -e   # Detiene el script si cualquier comando falla

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${REPO_DIR}/ros2_ws"
AUTOSTART_DIR="${HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/panel_kiosco.desktop"
UDEV_RULE="/etc/udev/rules.d/99-gpio.rules"

echo "=============================================="
echo "  Instalador Modo Kiosco — Panel NEMA 17"
echo "  Repositorio: ${REPO_DIR}"
echo "=============================================="
echo ""

# ── Paso 1: Regla udev para GPIO sin sudo ─────────────────────────────────
echo "[1/3] Configurando regla udev para GPIO..."

echo 'SUBSYSTEM=="gpio", MODE="0666"' | sudo tee "${UDEV_RULE}" > /dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "  ✓ Regla escrita en ${UDEV_RULE}"
echo "  ✓ Reglas udev recargadas"
echo ""

# ── Paso 2: Directorio de autostart ───────────────────────────────────────
echo "[2/3] Creando directorio de autostart GNOME..."

mkdir -p "${AUTOSTART_DIR}"
echo "  ✓ ${AUTOSTART_DIR} listo"
echo ""

# ── Paso 3: Archivo .desktop de autostart ─────────────────────────────────
echo "[3/3] Generando entrada de autostart: ${DESKTOP_FILE}"

cat > "${DESKTOP_FILE}" << EOF
[Desktop Entry]
Type=Application
Name=PanelKioscoROS2
Comment=Panel de control NEMA 17 - arranca automáticamente con GNOME
Exec=bash -c "sleep 8 && export DISPLAY=:0 && source /opt/ros/jazzy/setup.bash && source ${WS_DIR}/install/setup.bash && ros2 launch stepper_control_pkg main_launch.py > ${HOME}/registro_kiosco.log 2>&1"
X-GNOME-Autostart-enabled=true
EOF

echo "  ✓ ${DESKTOP_FILE} generado"
echo ""

# ── Resumen ────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Instalación completada."
echo ""
echo "  PRÓXIMOS PASOS:"
echo "  1. Compilar el workspace si no lo has hecho:"
echo "     cd ${WS_DIR} && colcon build"
echo "     source ${WS_DIR}/install/setup.bash"
echo ""
echo "  2. Reiniciar la sesión de GNOME o el sistema"
echo "     para que el autostart surta efecto."
echo ""
echo "  3. El sistema arrancará automáticamente"
echo "     5 segundos después de iniciar sesión."
echo "=============================================="
