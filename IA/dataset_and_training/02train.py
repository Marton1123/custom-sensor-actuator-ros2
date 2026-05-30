#!/usr/bin/env python3
"""
02_train.py  (v2 — bottle vs can, 2 clases)
============================================
Entrena YOLOv8n con 2 clases (bottle=0, can=1) y exporta a NCNN.

Hardware objetivo: servidor con 2x RTX A5000 (24 GB VRAM c/u) + 64 núcleos.
Restricciones de concurrencia aplicadas:
    - num_workers = 8   (no escalar a los 64 núcleos; protege /dev/shm y RAM)
    - torch.set_num_threads(4)  (evita que PyTorch acapare hilos del sistema)
    - os.environ OMP/MKL limitados a 4  (evita explosión de hilos en numpy/OpenCV)

Uso:
    conda activate botellas
    python scripts/02_train.py
"""

import os
import torch
from pathlib import Path
import shutil

# ─── Limitar hilos ANTES de importar Ultralytics/OpenCV ──────────────────────
# Evita que PyTorch, OpenMP y MKL acaparen los 64 núcleos del servidor
# causando congelamiento sistémico por explosión de hilos.
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["MKL_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"]    = "4"
torch.set_num_threads(4)
# ──────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO  # noqa: E402 — importar después de fijar hilos

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "raw" / "data.yaml"
RUNS_DIR  = BASE_DIR / "runs" / "train"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hiperparámetros — servidor (2x RTX A5000, 64 núcleos) ───────────────────
RUN_NAME = "botellas_vs_latas"
EPOCHS   = 100
IMG_SIZE = 640
BATCH    = 64      # 64 imgs / 2 GPUs = 32 por GPU — holgado para 24 GB VRAM
WORKERS  = 8       # máximo seguro; NO usar los 64 núcleos (protege /dev/shm)
DEVICE   = [0, 1]  # DDP dual GPU
PATIENCE = 15      # early stopping

# Para entrenar en local (GTX 1650 4 GB), comentar arriba y descomentar:
# BATCH   = 16
# WORKERS = 4
# DEVICE  = 0
# ──────────────────────────────────────────────────────────────────────────────


def train() -> Path:
    print("=" * 58)
    print("  Paso 2a — Entrenamiento YOLOv8n (bottle vs can)")
    print("=" * 58)
    print(f"  torch threads : {torch.get_num_threads()}")
    print(f"  OMP threads   : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  workers       : {WORKERS}")
    print(f"  device        : {DEVICE}")
    print(f"  batch         : {BATCH}")

    assert DATA_YAML.exists(), (
        f"No encontré data.yaml en {DATA_YAML}\n"
        "Corre primero: python scripts/01_download_and_prepare.py"
    )

    yaml_txt = DATA_YAML.read_text()
    assert "nc: 2" in yaml_txt, (
        "data.yaml tiene nc != 2. Asegúrate de haber corrido "
        "la versión v2 de 01_download_and_prepare.py"
    )

    model = YOLO("yolov8n.pt")

    model.train(
        data     = str(DATA_YAML),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        workers  = WORKERS,
        device   = DEVICE,
        project  = str(RUNS_DIR),
        name     = RUN_NAME,
        exist_ok = True,
        patience = PATIENCE,
        # Augmentación estándar
        hsv_h  = 0.015,
        hsv_s  = 0.7,
        hsv_v  = 0.4,
        fliplr = 0.5,
        mosaic = 1.0,
        mixup  = 0.1,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    assert best.exists(), (
        f"No se generó best.pt en {best}.\n"
        f"Revisa logs en {RUNS_DIR / RUN_NAME}/"
    )
    print(f"\n[INFO] Entrenamiento completo. Mejor modelo: {best}")
    return best


def export_ncnn(weights: Path) -> Path:
    print("\n" + "=" * 58)
    print("  Paso 2b — Exportación NCNN (FP16)")
    print("=" * 58)

    model = YOLO(str(weights))
    export_path = model.export(
        format = "ncnn",
        imgsz  = IMG_SIZE,
        int8   = False,   # FP16 estable; INT8 requiere calibración extra
    )

    src  = Path(export_path)
    dest = MODEL_DIR / "botellas_vs_latas_ncnn"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    print(f"\n[INFO] Modelo NCNN guardado en: {dest}")
    for f in sorted(dest.iterdir()):
        if not f.name.startswith("__"):
            print(f"         {f.name:<35} {f.stat().st_size / 1024:>8.1f} KB")

    return dest


def main() -> None:
    weights  = train()
    ncnn_dir = export_ncnn(weights)

    print("\n" + "=" * 58)
    print("  ✅  Pipeline Paso 2 v2 completado")
    print("=" * 58)
    print(f"  PyTorch : runs/train/{RUN_NAME}/weights/best.pt")
    print(f"  NCNN    : models/botellas_vs_latas_ncnn/")
    print("=" * 58)
    print("\n  Copiar a RPi:")
    print(f"  scp -r {ncnn_dir} pi@<IP_RPI>:~/ros2_ws/models/")


if __name__ == "__main__":
    main()