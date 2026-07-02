# Sistema Modular de Clasificación Autónoma — ROS 2 Jazzy

Sistema embebido de visión computacional y control de hardware diseñado para operar sobre **Raspberry Pi 5**. El sistema orquesta inferencia Edge-AI, adquisición de datos sensoriales y control de actuadores mecatrónicos a través del middleware **ROS 2 Jazzy Jalisco**.

---

## Descripción General

El paquete `stepper_control_pkg` implementa una arquitectura modular de responsabilidad única distribuida en cuatro nodos ROS 2 independientes. La lógica de inferencia se ejecuta sobre el motor **NCNN** (redes convolucionales sobre CPU sin dependencia de CUDA), minimizando la huella de memoria y la latencia en sistemas embebidos. La interfaz HMI se desarrolló sobre **PyQt6**, operando como máquina de estados finitos.

---

## Arquitectura de Hardware

| Componente | Modelo / Estándar | Interfaz |
|---|---|---|
| Unidad de cómputo | Raspberry Pi 5 (8 GB) | — |
| Cámara | Cámara UVC compatible (p. ej. 640×480 @ 30 FPS) | USB / V4L2 `/dev/video0` |
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

| Nodo | Paquete | Archivo | Descripción |
|---|---|---|---|
| `nodo_camara` | `stepper_control_pkg` | `nodo_camara.py` | Captura fotogramas desde la cámara UVC y los publica de forma continua para minimizar latencia. |
| `nodo_vision` | `ros2_vision_pkg` | `nodo_vision.py` | Inferencia YOLOv8 NCNN (2 clases: can/bottle), máquina de estados del proyecto y visualización del veredicto. *(Nota: La verificación de botellas sucias está desactivada temporalmente).* |
| `nodo_gui` | `stepper_control_pkg` | `nodo_gui.py` | Interfaz gráfica HMI en PyQt6 para visualizar la cámara procesada, estado y control. |
| `nodo_balanza` | `stepper_control_pkg` | `nodo_balanza.py` | Interfaz con la celda de carga HX711 para pesar los envases. |
| `nodo_actuadores` | `stepper_control_pkg` | `nodo_actuadores.py` | Control de giro de los motores DC mediante puente H L298N. Alterna la dirección según si es lata (Dirección 2) o botella (Dirección 1). |

### Tópicos de Comunicación (Pub/Sub)

| Tópico | Tipo de Mensaje | QoS | Publicador | Suscriptor(es) |
|---|---|---|---|---|
| `/camara/video_raw` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_camara` | `nodo_vision` |
| `/camara/video_procesado` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_vision` | `nodo_gui` |
| `/camara/video_segmentado` | `sensor_msgs/Image` | BEST\_EFFORT, depth=1 | `nodo_vision` | `nodo_gui` |
| `/clasificacion_objeto` | `std_msgs/String` | RELIABLE, depth=10 | `nodo_vision` | `nodo_gui` |
| `/tamano_estimado` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_vision` | `nodo_gui` |
| `/comando_grados` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_vision` / `nodo_gui` | `nodo_actuadores` |
| `/peso_elemento` | `std_msgs/Float32` | RELIABLE, depth=10 | `nodo_balanza` | `nodo_gui` |

### Diagrama de Flujo de Datos

```
[Cámara UVC] ──▶ nodo_camara ──▶ /camara/video_raw ──▶ nodo_vision ──▶ /camara/video_procesado  ──▶┐
                                                            │      ──▶ /camara/video_segmentado ──▶│
                                                            │      ──▶ /clasificacion_objeto    ──▶│── nodo_gui ──▶ [Pantalla HMI]
                                                            │      ──▶ /tamano_estimado         ──▶│
                                                            └──────▶ /comando_grados ──▶┐          │
                                                                                        ▼          │
[HX711] ───────▶ nodo_balanza ──▶ /peso_elemento ──────────────────────────────────────────────────┘
                                                                           nodo_actuadores ──▶ [Motores L298N]
```


---

## Requisitos del Sistema

### Dependencias del Sistema Operativo

```bash
sudo apt install python3-lgpio python3-gpiozero
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
ros2 launch stepper_control_pkg main_launch.py
```

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

### Scripts Utilitarios de Calibración y Lanzamiento

* **Ejecutar el proyecto en la Raspberry Pi:**
  ```bash
  /home/lab-ros/ejecutar_proyecto.sh
  ```
  *(Este script detiene ejecuciones previas colgadas, configura la orientación vertical de la pantalla y lanza todo el ecosistema de ROS 2).*

* **Calibrar el giro y dirección de los motores:**
  ```bash
  # Probar giro por 3.7 segundos en Dirección 1 (Dirección para Botellas)
  python3 calibrar_dos_motores.py 3.7 1

  # Probar giro por 3.7 segundos en Dirección 2 (Dirección para Latas)
  python3 calibrar_dos_motores.py 3.7 2
  ```

* **Detener motores inmediatamente en caso de emergencia:**
  ```bash
  python3 detener_motor.py
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
│       └── botellas_vs_latas_ncnn/    # Modelo exportado a NCNN (.bin + .param)
├── ros2_ws/
│   └── src/
│       ├── ros2_vision_pkg/           # Paquete de visión de ROS 2
│       │   ├── ros2_vision_pkg/
│       │   │   ├── nodo_vision.py     # Nodo de inferencia YOLOv8 y máquina de estados
│       │   │   └── segmentacion_respaldo.py # Respaldo del segmentador HSV (botella sucia)
│       │   └── package.xml
│       └── stepper_control_pkg/       # Paquete de control y GUI de ROS 2
│           ├── config/
│           │   └── parametros.yaml    # Parámetros YAML del sistema
│           ├── launch/
│           │   └── main_launch.py     # Orquestador de lanzamiento (5 nodos)
│           ├── stepper_control_pkg/   # Nodos Python de ROS 2
│           │   ├── nodo_actuadores.py # Control de dirección de motores L298N
│           │   ├── nodo_balanza.py    # Peso HX711
│           │   ├── nodo_camara.py     # Publicador de frames (bajo consumo)
│           │   └── nodo_gui.py        # Dashboard visual PyQt6
│           └── package.xml
├── calibrar_dos_motores.py            # Utilitario para pruebas de giro y sentido de motores
├── calibrar_tiempo.py                 # Utilitario para calibrar tiempos de ejecución
├── detener_motor.py                   # Script de apagado de emergencia de motores
├── ejecutar_proyecto.sh               # Script principal de lanzamiento del kiosco
├── ejecutar_proyecto.desktop         # Acceso directo para escritorio GNOME
└── README.md
```
