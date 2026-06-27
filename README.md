# Sistema Modular de Clasificación Autónoma — ROS 2 Jazzy

Sistema embebido de visión computacional y control de hardware diseñado para operar sobre **Raspberry Pi 5**. El sistema orquesta inferencia Edge-AI, adquisición de datos sensoriales y control de actuadores mecatrónicos a través del middleware **ROS 2 Jazzy Jalisco**.

---

## Descripción General

El sistema implementa una arquitectura modular distribuida en nodos ROS 2
independientes. La UnitV K210 se utiliza como cámara USB-serie y la lógica de
inferencia continúa ejecutándose sobre **NCNN** en la Raspberry Pi. La interfaz
HMI se desarrolló sobre **PyQt6**, operando como máquina de estados finitos.

---

## Arquitectura de Hardware

| Componente | Modelo / Estándar | Interfaz |
|---|---|---|
| Unidad de cómputo | Raspberry Pi 5 (8 GB) | — |
| Cámara | M5Stack UnitV K210/OV7740 | USB-serie (FTDI) |
| Sensor de masa | Celda de carga + convertidor HX711 (24-bit ADC) | GPIO bit-banging (gpiozero) |
| Motor paso a paso | NEMA 17 + controlador pulso/dirección (p. ej. TB6600) | GPIO lgpio — PUL/DIR |

### Pinout GPIO (BCM numbering — Raspberry Pi 5)

| Señal | Pin BCM | Descripción |
|---|---|---|
| `PUL+` (Step) | GPIO 17 | Pulso de paso — motor paso a paso |
| `DIR+` (Direction) | GPIO 27 | Dirección de giro (CW / CCW) |
| `DT` (HX711 Data) | GPIO 5 | Línea de datos del ADC de 24 bits |
| `SCK` (HX711 Clock) | GPIO 6 | Reloj de sincronización del ADC |

---

## Arquitectura de Software (ROS 2)

### Nodos del Sistema

| Nodo | Archivo | Descripción |
|---|---|---|
| `nodo_camara` | `nodo_camara.py` | Captura UVC antigua, conservada como alternativa |
| `nodo_k210_serial` | `nodo_k210_serial.py` | Recibe el video JPEG de la UnitV por USB |
| `nodo_vision` | `nodo_vision.py` | FSM de clasificación con backend NCNN o K210 |
| `nodo_gui` | `nodo_gui.py` | Dashboard HMI PyQt6 — visualización pasiva de todos los tópicos |
| `nodo_balanza` | `nodo_balanza.py` | Lectura del sensor HX711, tara automática, publicación de masa |
| `nodo_actuadores` | `nodo_actuadores.py` | Control GPIO del motor paso a paso vía lgpio |

### Tópicos de Comunicación (Pub/Sub)

| Tópico | Tipo de Mensaje | QoS | Publicador | Suscriptor(es) |
|---|---|---|---|---|
| `/camara/video_raw` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_k210_serial` | `nodo_vision` |
| `/camara/video_procesado` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_vision` | `nodo_gui` |
| `/camara/video_segmentado` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_vision` | `nodo_gui` |
| `/clasificacion_objeto` | `std_msgs/String` | RELIABLE, depth=10 | `nodo_vision` | `nodo_gui` |
| `/tamano_estimado` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_vision` | `nodo_gui` |
| `/peso_elemento` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_balanza` | `nodo_gui` |
| `/comando_grados` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_gui` *(externo)* | `nodo_actuadores` |
| `/comando_motor` | `std_msgs/Int32` | RELIABLE, depth=10 | *(externo)* | `nodo_actuadores` |

### Diagrama de Flujo de Datos

```
[UnitV/OV7740] ─USB─▶ nodo_k210_serial ─▶ /camara/video_raw
                                                   │
                                                   ▼
                                      nodo_vision (YOLO/NCNN en Pi)
                                                   │
                              /camara/video_procesado
                              /camara/video_segmentado
                              /clasificacion_objeto ───────▶ nodo_gui ─▶ [Pantalla HMI]

[HX711] ─▶ nodo_balanza ─▶ /peso_elemento ─────────────────▶ nodo_gui
[nodo_gui / CLI] ─▶ /comando_grados ─▶ nodo_actuadores ────▶ [Motor]
```

---

## Requisitos del Sistema

### Dependencias del Sistema Operativo

```bash
sudo apt install python3-lgpio python3-gpiozero python3-serial
sudo apt install ros-jazzy-cv-bridge ros-jazzy-sensor-msgs
pip install ncnn PyQt6
```

> **Permisos GPIO:** El usuario debe pertenecer al grupo `gpio` para acceder a `/dev/gpiochip4`:
> ```bash
> sudo usermod -aG gpio $USER
> ```

### Dependencias Python (declaradas en `package.xml`)

- `rclpy`, `std_msgs`, `sensor_msgs`, `cv_bridge`
- `opencv-python`, `ncnn`, `numpy`
- `PyQt6`, `gpiozero`, `lgpio`

---

## Instalación y Compilación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio> ~/custom-sensor-actuator-ros2
cd ~/custom-sensor-actuator-ros2

# 2. Fuente del entorno ROS 2
source /opt/ros/jazzy/setup.bash

# 3. Compilar el workspace
cd ros2_ws
colcon build --symlink-install

# 4. Aplicar el entorno compilado
source install/setup.bash
```

---

## Uso

### Lanzamiento del Sistema Completo

```bash
# UnitV K210 como cámara + YOLOv8/NCNN en Raspberry
export DISPLAY=:0
ros2 launch stepper_control_pkg main_launch.py \
  serial_port:=/dev/ttyUSB0 baudrate:=115200

# Alternativa futura: YOLOv2 cuantizado ejecutado en la K210
ros2 launch stepper_control_pkg k210_system.launch.py
```

La preparación del dataset, el firmware MaixPy, el cableado UART y el
entrenamiento cuantizado para K210 se documentan en
[`IA/k210/README.md`](IA/k210/README.md).

### Ajuste de Parámetros en Tiempo de Ejecución

```bash
# Modificar velocidad del motor sin reiniciar el nodo
ros2 param set /nodo_actuadores delay_pulso 0.001

# Enviar comando de rotación manual (en grados)
ros2 topic pub --once /comando_grados std_msgs/Float32 "data: 90.0"

# Enviar pasos directos al motor
ros2 topic pub --once /comando_motor std_msgs/Int32 "data: 800"
```

### Monitoreo en Tiempo Real

```bash
# Ver clasificación del objeto detectado
ros2 topic echo /clasificacion_objeto

# Ver peso del sensor de carga
ros2 topic echo /peso_elemento

# Ver área transversal estimada
ros2 topic echo /tamano_estimado
```

---

## Configuración de Parámetros (YAML)

El archivo `config/parametros.yaml` centraliza los parámetros cargados por los nodos en el lanzamiento:

```yaml
nodo_balanza:
  ros__parameters:
    hx711_reference_unit: 2273.9   # Factor de calibración ADC→gramos

nodo_camara:
  ros__parameters:
    distancia_camara_cm: 60.0      # Distancia focal de referencia para estimación de área
```

### Calibración del Sensor de Masa

Cada celda de carga física tiene derivaciones estructurales únicas. Para calibrar:

1. Abrir `nodo_balanza.py` y localizar el parámetro `hx711_reference_unit` en el archivo YAML.
2. Colocar una masa conocida sobre el sensor.
3. Ajustar el valor de `hx711_reference_unit` hasta que la lectura publicada en `/peso_elemento` coincida con la masa real.
4. Recompilar con `colcon build` para persistir el cambio.

---

## Estructura del Repositorio

```
custom-sensor-actuator-ros2/
├── IA/
│   ├── dataset_and_training/          # Scripts de entrenamiento y datasets
│   └── models/
│       └── botellas_ncnn_model/       # Modelo exportado a NCNN (.bin + .param)
├── ros2_ws/
│   └── src/
│       └── stepper_control_pkg/
│           ├── config/
│           │   └── parametros.yaml    # Parámetros YAML del sistema
│           ├── launch/
│           │   └── main_launch.py     # Orquestador de producción
│           ├── stepper_control_pkg/   # Ejecutables Python (Nodos ROS 2)
│           │   ├── nodo_actuadores.py
│           │   ├── nodo_balanza.py
│           │   ├── nodo_camara.py
│           │   └── nodo_gui.py
│           ├── package.xml            # Manifiesto de dependencias ROS 2
│           └── setup.py              # Registro de entry points
└── README.md
```
