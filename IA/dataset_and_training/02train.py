#!/usr/bin/env python3
"""
02_train.py  (v2 — bottle vs can, 2 clases)
============================================
Entrena YOLOv8n con 2 clases (bottle=0, can=1) y exporta a NCNN.

Dataset real tras descarga:
    ~12200 instancias bottle  vs  ~2600 instancias can  (ratio ~4.7:1)
Estrategia anti-desbalance:
    - cls_weight aumentado (penaliza más errores en la clase minoritaria)
    - copy_paste + degrees para aumentar variedad visual de latas

Hardware objetivo: servidor con 2x RTX A5000 (24 GB VRAM c/u) + 64 núcleos.
Restricciones de concurrencia:
    - num_workers = 8   (protege /dev/shm y RAM compartida)
    - torch/OMP/MKL limitados a 4 hilos (evita explosión sistémica)

Uso:
    conda activate botellas
    python scripts/02_train.py
"""

import os
import shutil
from pathlib import Path

import torch

# ─── Limitar hilos ANTES de importar Ultralytics/OpenCV ──────────────────────
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["MKL_NUM_THREADS"]        = "4"
os.environ["OPENBLAS_NUM_THREADS"]   = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"]    = "4"
torch.set_num_threads(4)
# ──────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO  # noqa: E402

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
BATCH    = 64      # 32 por GPU — holgado para 24 GB VRAM
WORKERS  = 8       # NO escalar a 64; protege /dev/shm
DEVICE   = [0, 1]  # DDP dual GPU
PATIENCE = 20      # más paciencia por dataset desbalanceado

# Para local (GTX 1650): BATCH=16, WORKERS=4, DEVICE=0
# ──────────────────────────────────────────────────────────────────────────────

# ─── Compensación de desbalance ───────────────────────────────────────────────
# Instancias reales tras descarga: ~12200 bottle vs ~2600 can (ratio ~4.7)
# cls controla el peso de la classification loss en YOLOv8.
# Subir cls hace que el modelo penalice más confundir clases → ayuda a la
# clase minoritaria (can). Valor base Ultralytics = 0.5.
# ratio 4.7 → cls = 0.5 * 4.7 ≈ 2.3  (redondeamos a 2.0 para no sobreajustar)
CLS_WEIGHT = 2.0
# ──────────────────────────────────────────────────────────────────────────────


def _get_imbalance_ratio() -> float:
    """Lee el data.yaml y estima el ratio de desbalance desde los labels."""
    lbl_train = BASE_DIR / "data" / "raw" / "labels" / "train"
    counts = {0: 0, 1: 0}
    for lbl in lbl_train.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] = counts.get(int(parts[0]), 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 1.0
    print(
        f"[INFO] Instancias train — bottle: {counts[0]} "
        f"({counts[0]/total*100:.1f}%)  |  "
        f"can: {counts[1]} ({counts[1]/total*100:.1f}%)"
    )
    ratio = counts[0] / max(counts[1], 1)
    print(f"[INFO] Ratio desbalance bottle/can: {ratio:.2f}  →  cls_weight={CLS_WEIGHT}")
    return ratio


def train() -> Path:
    print("=" * 60)
    print("  Paso 2a — Entrenamiento YOLOv8n (bottle vs can)")
    print("=" * 60)
    print(f"  torch threads : {torch.get_num_threads()}")
    print(f"  OMP threads   : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  workers       : {WORKERS}")
    print(f"  device        : {DEVICE}")
    print(f"  batch         : {BATCH}")

    assert DATA_YAML.exists(), (
        f"No encontré data.yaml en {DATA_YAML}\n"
        "Corre primero: python scripts/01_download_and_prepare.py"
    )
    assert "nc: 2" in DATA_YAML.read_text(), (
        "data.yaml tiene nc != 2. Asegúrate de haber corrido "
        "la versión v2 de 01_download_and_prepare.py"
    )

    _get_imbalance_ratio()

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

        # ── Augmentación ──────────────────────────────────────────
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.15,   # subido ligeramente para mezclar más latas
        copy_paste = 0.3,    # copia objetos de latas entre imágenes → +variedad
        degrees    = 10.0,   # rotación leve → latas en distintas orientaciones

        # ── Compensación desbalance ────────────────────────────────
        # cls_weight global: penaliza más los errores de clasificación
        # para que la clase minoritaria (can) no sea ignorada.
        cls        = CLS_WEIGHT,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    assert best.exists(), (
        f"No se generó best.pt en {best}.\n"
        f"Revisa logs en {RUNS_DIR / RUN_NAME}/"
    )
    print(f"\n[INFO] Entrenamiento completo. Mejor modelo: {best}")
    return best


def export_ncnn(weights: Path) -> Path:
    print("\n" + "=" * 60)
    print("  Paso 2b — Exportación NCNN (FP16)")
    print("=" * 60)

    model = YOLO(str(weights))
    export_path = model.export(
        format = "ncnn",
        imgsz  = IMG_SIZE,
        int8   = False,
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

    print("\n" + "=" * 60)
    print("  ✅  Pipeline Paso 2 v2 completado")
    print("=" * 60)
    print(f"  PyTorch : runs/train/{RUN_NAME}/weights/best.pt")
    print(f"  NCNN    : models/botellas_vs_latas_ncnn/")
    print("=" * 60)
    print("\n  Copiar a RPi:")
    print(f"  scp -r {ncnn_dir} pi@<IP_RPI>:~/ros2_ws/models/")


if __name__ == "__main__":
    main()