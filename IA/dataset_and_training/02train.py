#!/usr/bin/env python3
"""
02_train.py
===========
Paso 2 del pipeline: entrena YOLOv8n con una sola clase (bottle)
sobre el dataset preparado en el paso anterior y exporta el modelo
a formato NCNN con cuantización INT8.

Flujo:
    1. Entrena YOLOv8n en GPU (GTX 1650 / CUDA)
    2. Valida automáticamente al final del entrenamiento
    3. Exporta el mejor checkpoint a NCNN INT8

Salida:
    runs/train/botellas/weights/best.pt     ← pesos PyTorch
    models/botellas_ncnn_model/
        ├── model.ncnn.param                ← arquitectura
        └── model.ncnn.bin                  ← pesos INT8

Uso:
    conda activate botellas
    python scripts/02_train.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "raw" / "data.yaml"
RUNS_DIR  = BASE_DIR / "runs" / "train"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
RUN_NAME   = "botellas"
EPOCHS     = 50
IMG_SIZE   = 640
BATCH      = 64      # seguro para GTX 1650 (4 GB VRAM) con 640px
WORKERS    = 8
DEVICE     = [0, 1]       # GPU 0. Cambiar a "cpu" solo para pruebas rápidas.
PATIENCE   = 15      # early stopping: detiene si no mejora en 15 epochs
# ──────────────────────────────────────────────────────────────────────────────


def train() -> Path:
    print("=" * 55)
    print("  Paso 2a — Entrenamiento YOLOv8n")
    print("=" * 55)

    assert DATA_YAML.exists(), f"No encontré data.yaml en {DATA_YAML}\n" \
                                "Asegúrate de haber corrido 01_download_and_prepare.py primero."

    model = YOLO("yolov8n.pt")  # descarga automática si no existe (~6 MB)

    results = model.train(
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        workers   = WORKERS,
        device    = DEVICE,
        project   = str(RUNS_DIR),
        name      = RUN_NAME,
        exist_ok  = True,
        patience  = PATIENCE,   # early stopping
        # Augmentación estándar (buena para objetos domésticos)
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    assert best.exists(), f"No se generó best.pt en {best}"
    print(f"\n[INFO] Entrenamiento completo.")
    print(f"[INFO] Mejor modelo: {best}")
    return best


def export_ncnn(weights: Path) -> Path:
    print("\n" + "=" * 55)
    print("  Paso 2b — Exportación NCNN INT8")
    print("=" * 55)
    print("[INFO] Esto puede tardar unos minutos (calibración INT8)...")

    model = YOLO(str(weights))
    export_path = model.export(
        format = "ncnn",
        imgsz  = IMG_SIZE,
        int8   = True,
        data   = str(DATA_YAML),   # imágenes de calibración para INT8
    )

    # Copiar a models/ con nombre limpio
    src  = Path(export_path)
    dest = MODEL_DIR / "botellas_ncnn_model"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    print(f"\n[INFO] Modelo NCNN INT8 guardado en: {dest}")
    print("[INFO] Archivos para desplegar en la RPi 5:")
    for f in sorted(dest.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"         {f.name:<30} {size_kb:>8.1f} KB")

    return dest


def print_summary(weights: Path, ncnn_dir: Path) -> None:
    print("\n" + "=" * 55)
    print("  ✅  Pipeline Paso 2 completado")
    print("=" * 55)
    print(f"  PyTorch : {weights.relative_to(BASE_DIR)}")
    print(f"  NCNN    : {ncnn_dir.relative_to(BASE_DIR)}/")
    print("=" * 55)
    print("\n  Siguiente paso: copiar la carpeta NCNN a la RPi 5")
    print(f"  scp -r {ncnn_dir} pi@<IP_RPI>:~/ros2_ws/models/")
    print("\n  Luego lanzar el nodo de visión:")
    print("  ros2 run nodo_vision nodo_vision")


def main() -> None:
    weights  = train()
    ncnn_dir = export_ncnn(weights)
    print_summary(weights, ncnn_dir)


if __name__ == "__main__":
    main()