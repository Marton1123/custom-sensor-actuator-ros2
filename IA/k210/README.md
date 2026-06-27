# UnitV K210: botella frente a lata

Esta variante conserva video enviando JPEG por el mismo cable USB usado para
alimentar la UnitV. En la Raspberry aparece como un puerto FTDI
(`/dev/ttyUSB0`), no como una cámara UVC. El modo compatible inicial utiliza
115200 bps y entrega aproximadamente 5 FPS con JPEG de unos 1,5 KB.

## Prueba inicial sin modelo

Antes de entrenar, se puede validar toda la ruta de cámara usando
`unitv_camera_only.py`. Este programa no necesita `.kmodel` ni anchors:

1. Grabar firmware MaixPy-v1 compatible con UnitV/M5StickV.
2. Desde VS Code, instalar el arranque y programa de cámara por el REPL
   normal:

   ```powershell
   python IA/k210/upload_maixpy.py --port COM5 --install-camera
   ```

   Si el programa de cámara ya está transmitiendo y no responde con `>>>`,
   desconectar la UnitV, iniciar el siguiente comando y volver a conectarla
   mientras espera:

   ```powershell
   python IA/k210/upload_maixpy.py --port COM5 --wait-for-port 30 --install-camera
   ```

   Si una versión anterior dejó el REPL a otra velocidad, recuperarlo e
   instalar en un solo comando:

   ```powershell
   python IA/k210/upload_maixpy.py --port COM5 --recover-from-baud 1500000 --install-camera
   ```

   El cargador respalda el demo original como
   `/flash/boot_m5stickv.py`, instala un `boot.py` mínimo y copia
   `unitv_camera_only.py` como `/flash/main.py`. El `boot.py` ejecuta
   explícitamente `main.py`, porque el firmware M5Stack v5.1.2 no lo inicia
   automáticamente.
3. Reiniciar la UnitV y validar el flujo desde Windows:

   ```powershell
   python IA/k210/test_usb_stream.py --port COM5
   ```

4. Conectar el USB-C de la UnitV a un puerto USB de la Raspberry.
5. Lanzar `k210_system.launch.py`.

En esta modalidad se publican `/camara/video_raw` y
`/camara/video_procesado`, mientras `/k210/detecciones` contiene una lista
vacía. Por lo tanto, la GUI muestra video pero no clasifica objetos.

## 1. Preparar el dataset en el servidor

El dataset actual ya está en formato YOLO:

- `IA/data/raw/images/{train,val}`
- `IA/data/raw/labels/{train,val}`
- clase `0`: `bottle`
- clase `1`: `can`

El entrenador oficial `sipeed/maix_train` consume Pascal VOC. Desde la raíz
del repositorio:

```bash
conda activate botellas
pip install Pillow
python IA/dataset_and_training/prepare_k210_dataset.py
```

El resultado es `IA/data/bottle_can_k210_voc.zip`.

## 2. Entrenar y cuantizar Tiny-YOLOv2

`maix_train` es el pipeline oficial para K210 que entrena YOLOv2 a 224x224 y
genera el `.kmodel` cuantizado junto con el código y los anchors:

```bash
cd ~/CODE
git clone https://github.com/sipeed/maix_train.git
cd maix_train
pip install -r requirements.txt
python train.py init
python train.py -t detector \
  -z ../custom-sensor-actuator-ros2/IA/data/bottle_can_k210_voc.zip train
```

Es un pipeline antiguo, probado oficialmente con TensorFlow 2.1 y una versión
0.1 de nncase. Conviene ejecutarlo en un entorno Conda separado del entorno
de YOLOv8. La salida aparece en `maix_train/out/`.

Copiar el `.kmodel` resultante a:

```text
/sd/bottle_can.kmodel
```

Copiar los anchors generados por el entrenamiento en `ANCHORS` dentro de
`unitv_bottle_can.py`. No se deben conservar los anchors de ejemplo.

## 3. Instalar el programa en la UnitV

Copiar `unitv_bottle_can.py` como `/flash/main.py`. Puede utilizarse el
cargador una vez detenido el programa actual:

```powershell
python IA/k210/upload_maixpy.py --port COM5 --local IA/k210/unitv_bottle_can.py --remote /flash/main.py
```

El programa espera:

- modelo de entrada 224x224;
- salida YOLOv2 con dos clases en orden `bottle`, `can`;
- MaixPy-v1;
- USB-serie/UARTHS a 115200 bps.

## 4. Conexión USB con Raspberry Pi

Conectar únicamente el USB-C de la UnitV a un puerto USB-A de la Raspberry.
No se utilizan el conector Grove ni los GPIO. El dispositivo normalmente
aparece como `/dev/ttyUSB0`; para identificarlo de forma estable:

```bash
ls -l /dev/serial/by-id/
```

No es necesario modificar `/boot/firmware/config.txt`. Sólo debe agregarse el
usuario a `dialout`:

```bash
sudo usermod -aG dialout "$USER"
```

Reiniciar después de cambiar el grupo.

## 5. Ejecutar ROS 2

```bash
sudo apt install python3-serial
cd ~/custom-sensor-actuator-ros2/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch stepper_control_pkg k210_system.launch.py
```

Verificar el dispositivo después de conectar la UnitV:

```bash
ls -l /dev/ttyUSB0
ros2 launch stepper_control_pkg k210_system.launch.py \
  serial_port:=/dev/ttyUSB0 baudrate:=115200
```

Si el enlace es inestable, cambiar `BAUDRATE` en la UnitV y `baudrate` en el
launch al mismo valor. Los valores superiores deben verificarse con el
adaptador FTDI y firmware concretos antes de aumentar los FPS.

## Limitación inicial del dataset

Las 7.394 imágenes actuales provienen de COCO/Open Images, no del OV7740. El
primer modelo sirve para validar el despliegue, pero su precisión real debe
medirse con la UnitV. Después conviene incorporar imágenes del montaje final
—misma distancia, fondo e iluminación— y reentrenar.
