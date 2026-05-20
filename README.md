# Arquitectura y Documentación Técnica - Proyecto Principal MVP

## Descripción Arquitectónica

El presente proyecto implementa la arquitectura fundacional de un sistema automatizado, diseñado para operar sobre sistemas embebidos (específicamente Raspberry Pi 5). El sistema orquesta visión artificial de borde (Edge AI), adquisición de datos sensoriales físicos y control de actuadores mecatrónicos a través del middleware ROS 2 (Jazzy Jalisco).

La lógica de inferencia computacional se ha desacoplado de frameworks de alto nivel como PyTorch en favor del motor NCNN, permitiendo ejecución de redes neuronales convolucionales sobre CPU con latencia estrictamente controlada y mínima huella de memoria (RAM). La interfaz humano-máquina (HMI) se encuentra desarrollada sobre PyQt6, operando como una máquina de estados finitos que gobierna el ciclo de vida transaccional del sistema.

## Topología de Nodos

El metapaquete `stepper_control_pkg` implementa una arquitectura modular de responsabilidades únicas distribuidas en los siguientes nodos de ROS 2:

*   **nodo_camara.py**: Responsable de la captura de flujo de video mediante Video4Linux2 (V4L2) operando en hilos asíncronos para evitar bloqueos por I/O. Integra inferencia mediante NCNN puro (decodificación de tensores, Letterboxing y Non-Maximum Suppression). Genera y publica cálculos analíticos de área de cajas delimitadoras (Bounding Boxes) bajo políticas de Quality of Service (QoS) diseñadas para anular latencia (`ReliabilityPolicy.BEST_EFFORT`, `HistoryPolicy.KEEP_LAST`).
*   **nodo_gui.py**: Instancia la interfaz gráfica interactiva del panel de usuario mediante PyQt6. Actúa como el orquestador lógico del sistema, evaluando métricas combinadas (tamaño visual frente a masa física) para emitir el dictamen final sobre la viabilidad de la recepción del envase (por ejemplo, rechazo por presencia de líquido remanente).
*   **nodo_balanza.py**: Interfaz de hardware para el circuito integrado HX711 (convertidor A/D de 24 bits). Desarrollado utilizando `gpiozero` para asegurar compatibilidad con la arquitectura del controlador perimetral RP1 presente en la Raspberry Pi 5. Incluye rutinas de calibración por software, tara automática (auto-tare) durante la fase de inicialización y manejo seguro de excepciones ante fallos de lectura diferencial.
*   **nodo_actuadores.py**: Responsable de traducir las sentencias de aceptación o rechazo lógico en pulsos electrónicos directos (manipulación GPIO a nivel de hardware) orientados al controlador del motor paso a paso, gestionando secuencias de rotación para aceptar contenedores hacia el depósito o expulsarlos hacia el usuario.

## Mapa de Tópicos (Topics)

La comunicación inter-procesos se define a través del siguiente bus de publicación/suscripción:

*   `/camara/video_raw` (`sensor_msgs/Image`): Transmisión en tiempo real de los fotogramas anotados a 30 FPS desde el nodo de visión a la interfaz gráfica.
*   `/analisis_botella` (`std_msgs/String`): Clasificación del tamaño detectado (e.g., "chica", "grande", "vacio").
*   `/tamano_estimado` (`std_msgs/Float32`): Valor físico estimado del área transversal capturada en cm² (determinado a partir del cociente pixel-espacio).
*   `/peso_botella` (`std_msgs/Float32`): Lectura en tiempo real de la masa del envase depositado en el receptáculo en gramos.
*   `/comando_grados` (`std_msgs/Float64`): Instrucción angular expedida por la máquina de estados hacia el motor paso a paso.

## Mapa de Hardware (Pinout RPi 5)

La integración electrónica directa hacia la placa base se debe realizar respetando el siguiente esquema de asignación GPIO (Broadcom BCM numbering):

*   **Motor Stepper (Controlador de Paso)**:
    *   `PUL` (Pulse): GPIO 17
    *   `DIR` (Direction): GPIO 27
*   **Balanza (Celda de Carga vía HX711)**:
    *   `DT` (Data): GPIO 5
    *   `SCK` (Clock): GPIO 6

## Estructura del Proyecto

```text
custom-sensor-actuator-ros2/
├── IA/                                   # Recursos de inteligencia artificial
│   ├── dataset_and_training/             # Scripts de entrenamiento y datasets crudos
│   └── models/
│       └── botellas_ncnn_model/          # Modelos exportados a formato NCNN (.bin, .param)
├── ros2_ws/                              # Espacio de trabajo ROS 2
│   └── src/
│       └── stepper_control_pkg/          # Metapaquete principal del proyecto
│           ├── launch/
│           │   └── main_launch.py        # Archivo maestro de despliegue
│           ├── stepper_control_pkg/      # Directorio de ejecutables Python (Nodos)
│           │   ├── nodo_actuadores.py
│           │   ├── nodo_balanza.py
│           │   ├── nodo_camara.py
│           │   └── nodo_gui.py
│           ├── package.xml               # Manifiesto de dependencias ROS 2
│           └── setup.py                  # Script de registro y compilación
└── README.md                             # Documentación arquitectónica
```

## Instrucciones de Despliegue

Para desplegar y ejecutar el entorno de producción en el hardware objetivo, ejecutar en terminal la siguiente secuencia operacional:

1. Sincronización del repositorio:
   ```bash
   cd ~/custom-sensor-actuator-ros2
   git pull origin test-yolo
   ```

2. Compilación del espacio de trabajo ROS 2:
   ```bash
   cd ros2_ws
   colcon build --symlink-install
   ```

3. Aplicación de variables de entorno y registro de paquetes:
   ```bash
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ```

4. Orquestación del sistema mediante Launch File:
   ```bash
   ros2 launch stepper_control_pkg main_launch.py
   ```

## Guía de Calibración

Las celdas de carga físicas poseen derivaciones estructurales únicas; por consiguiente, el convertidor análogo-digital requerirá ajuste del parámetro escalar base.

Para calibrar una nueva balanza física:
1. Abra el archivo `ros2_ws/src/stepper_control_pkg/stepper_control_pkg/nodo_balanza.py`.
2. Ubique la rutina del constructor (`__init__`) e identifique el atributo `self._reference_unit`.
3. Modifique el valor numérico (actualmente establecido en `1.0`) por el cociente de corrección correspondiente determinado empíricamente durante las pruebas con masa conocida.
4. Efectúe una recompilación del paquete mediante `colcon build` para aplicar la modificación permanente en el registro del nodo.
