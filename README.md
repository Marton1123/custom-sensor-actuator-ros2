# stepper_control_pkg — ROS 2 Jazzy · Raspberry Pi 5

Control de motor paso a paso **NEMA 17** mediante driver **TB6600** y lectura de sensor analógico/digital customizado, sobre **Ubuntu 24.04** y **ROS 2 Jazzy Jalisco**.

---

## Árbol del workspace

```
~/ros2_ws/
├── src/
│   └── stepper_control_pkg/
│       ├── stepper_control_pkg/
│       │   ├── __init__.py
│       │   ├── nodo_sensores.py       # Publica en /estado_sensor
│       │   └── nodo_actuadores.py     # Se suscribe y controla el motor
│       ├── launch/
│       │   └── stepper_system.launch.py
│       ├── config/
│       │   └── params.yaml
│       ├── resource/
│       │   └── stepper_control_pkg    # Marcador ament
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
└── (build/ install/ log/ — generados por colcon)
```

---

## Pines TB6600 → Raspberry Pi 5 (BCM, configurables)

| Señal TB6600 | Pin BCM RPi5 | Parámetro ROS 2 |
|:---:|:---:|:---:|
| STEP | 17 | `pin_step` |
| DIR  | 27 | `pin_dir`  |
| ENA  | 22 | `pin_ena`  |
| GND  | GND | — |

---

## Tópicos

| Tópico | Tipo | Productor | Consumidor |
|---|---|---|---|
| `/estado_sensor` | `std_msgs/Float32` | `nodo_sensores` | `nodo_actuadores` |
| `/control_manual` | `std_msgs/Bool` | externo | `nodo_actuadores` |

---

## Compilar y ejecutar

```bash
cd ~/ros2_ws
colcon build --packages-select stepper_control_pkg
source install/setup.bash

# Lanzar ambos nodos:
ros2 launch stepper_control_pkg stepper_system.launch.py

# O por separado:
ros2 run stepper_control_pkg nodo_sensores
ros2 run stepper_control_pkg nodo_actuadores

# Verificar tópicos:
ros2 topic echo /estado_sensor
ros2 topic pub /control_manual std_msgs/msg/Bool "data: true" --once
```

---

## Pipeline de dataset en PC (Windows/Linux, sin ROS)

Todo el flujo de dataset/entrenamiento ocurre en tu PC local. La Raspberry Pi solo ejecuta inferencia con el modelo final.

### 1) Captura + filtrado de frames + subida opcional a Roboflow

```bash
pip install opencv-python ultralytics roboflow

python IA/dataset_and_training/dataset_pipeline.py run ^
  --duration 180 ^
  --out-root .\dataset_work ^
  --upload ^
  --roboflow-api-key TU_KEY ^
  --roboflow-workspace TU_WORKSPACE ^
  --roboflow-project TU_PROYECTO
```

Este comando:
- graba video de camara,
- toma muestras de frames,
- conserva los que tienen deteccion de `bottle` (YOLOv8n COCO),
- y los sube a Roboflow en splits train/valid/test.

### 2) Entrenamiento + exportacion NCNN INT8 (PC)

```bash
python IA/dataset_and_training/train_and_export.py \
  --roboflow-api-key TU_KEY \
  --roboflow-workspace TU_WORKSPACE \
  --roboflow-project TU_PROYECTO \
  --roboflow-version 1
```

### 3) Despliegue a Raspberry Pi (solo modelo)

```bash
scp -r runs/detect/botella_yolov8n/weights/best_ncnn_model/ pi@<IP_RPI>:~/modelos/
```
