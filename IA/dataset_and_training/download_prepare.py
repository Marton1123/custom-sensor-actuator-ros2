#!/usr/bin/env python3
"""
01_download_and_prepare.py
==========================
Paso 1 del pipeline: descarga las imágenes de COCO que contienen botellas
(clase 39) usando FiftyOne, toma una muestra de 6000 imágenes (4800 train /
1200 val) y las convierte al formato YOLO normalizado.

Estructura de salida:
    data/
    └── raw/
        ├── images/
        │   ├── train/   ← 4800 imágenes .jpg
        │   └── val/     ← 1200 imágenes .jpg
        ├── labels/
        │   ├── train/   ← 4800 archivos .txt (formato YOLO)
        │   └── val/     ← 1200 archivos .txt
        └── data.yaml    ← config para Ultralytics

Uso:
    conda activate botellas
    python scripts/01_download_and_prepare.py

Tiempo estimado: 10–25 min según conexión (~2–3 GB de descarga).
"""

import random
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# ─── Configuración ────────────────────────────────────────────────────────────
COCO_CLASS    = "bottle"
YOLO_CLASS_ID = 0           # remap clase 39 → 0 (única clase del modelo)

N_TRAIN = 4800
N_VAL   = 1200
SEED    = 42

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
ZOO_DIR  = BASE_DIR / "data" / "_fiftyone_zoo"
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


def load_split(split: str, max_samples: int) -> fo.Dataset:
    """
    Carga desde FiftyOne Zoo solo las imágenes de COCO que contienen
    botellas. Si ya están en caché local, no vuelve a descargar.
    """
    print(f"\n[INFO] Cargando split '{split}' (máx. {max_samples} imgs con botellas)...")

    # Configurar directorio de caché ANTES de llamar al zoo
    # (evita el conflicto 'multiple values for dataset_dir' en FiftyOne >= 0.23)
    fo.config.dataset_zoo_dir = str(ZOO_DIR)

    dataset_name = f"coco_bottles_{split}"

    if fo.dataset_exists(dataset_name):
        print(f"[INFO] Dataset '{dataset_name}' ya en caché, cargando...")
        return fo.load_dataset(dataset_name)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=[COCO_CLASS],
        only_matching=True,
        max_samples=max_samples,
        dataset_name=dataset_name,
        seed=SEED,
    )
    print(f"[INFO] Imágenes cargadas en '{split}': {len(dataset)}")
    return dataset


def export_to_yolo(dataset: fo.Dataset, split_name: str) -> tuple[int, int]:
    """
    Copia imágenes y genera etiquetas YOLO normalizadas en data/raw/.
    Devuelve (exportadas, omitidas).
    """
    img_out = RAW_DIR / "images" / split_name
    lbl_out = RAW_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped  = 0

    for sample in dataset:
        if sample.ground_truth is None:
            skipped += 1
            continue

        bottle_dets = [
            d for d in sample.ground_truth.detections
            if d.label == COCO_CLASS
        ]
        if not bottle_dets:
            skipped += 1
            continue

        src = Path(sample.filepath)
        dst = img_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

        # FiftyOne bbox: [x_min_rel, y_min_rel, w_rel, h_rel]
        # YOLO bbox:     [x_center_rel, y_center_rel, w_rel, h_rel]
        lines = []
        for det in bottle_dets:
            x, y, w, h = det.bounding_box
            x_center = max(0.0, min(1.0, x + w / 2.0))
            y_center = max(0.0, min(1.0, y + h / 2.0))
            w        = max(0.001, min(1.0, w))
            h        = max(0.001, min(1.0, h))
            lines.append(
                f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        lbl_file = lbl_out / src.with_suffix(".txt").name
        lbl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1

    return exported, skipped


def compensate_val(ds_train: fo.Dataset, ds_val: fo.Dataset) -> tuple[fo.Dataset, fo.Dataset]:
    """
    Si val tiene menos de N_VAL muestras, toma el déficit de train.
    Usa una vista filtrada en vez de merge para evitar errores de la API.
    """
    real_val = len(ds_val)
    if real_val >= N_VAL:
        return ds_train, ds_val

    deficit = N_VAL - real_val
    print(f"\n[WARN] val solo tiene {real_val} imgs con botellas en COCO.")
    print(f"       Se usarán {deficit} imágenes extras de train para completar {N_VAL}.")

    all_train_ids = ds_train.values("id")
    random.shuffle(all_train_ids)
    extra_ids = all_train_ids[:deficit]
    remaining_ids = all_train_ids[deficit:]

    # Exportar el bloque extra directamente como si fuera val
    # (no mezclamos datasets, trabajamos con vistas)
    ds_extra    = ds_train.select(extra_ids)
    ds_train_v2 = ds_train.select(remaining_ids)

    print(f"[INFO] Tras ajuste → train: {len(ds_train_v2)}, val efectivo: {real_val + len(ds_extra)}")
    return ds_train_v2, ds_val, ds_extra   # type: ignore[return-value]


def write_data_yaml(n_train: int, n_val: int) -> None:
    yaml_text = f"""\
# ── Dataset: COCO Bottles (1 clase) ──────────────────────────
# Generado por 01_download_and_prepare.py
# train: {n_train} imágenes  |  val: {n_val} imágenes

path: {RAW_DIR.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: bottle
"""
    out = RAW_DIR / "data.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    print(f"\n[INFO] data.yaml escrito en: {out}")


def print_summary(n_train: int, n_val: int) -> None:
    print("\n" + "=" * 55)
    print("  ✅  Pipeline Paso 1 completado")
    print("=" * 55)
    print(f"  Train : {n_train:>5} imágenes  →  data/raw/images/train/")
    print(f"  Val   : {n_val:>5} imágenes  →  data/raw/images/val/")
    print(f"  Labels: formato YOLO normalizado  (clase 0 = bottle)")
    print(f"  YAML  : data/raw/data.yaml")
    print("=" * 55)
    print("\n  Siguiente paso:")
    print("  python scripts/02_train.py")


def main() -> None:
    print("=" * 55)
    print("  Pipeline Paso 1 — Descarga y preparación COCO Bottles")
    print("=" * 55)

    # ── 1. Descargar desde FiftyOne Zoo ───────────────────────────────────────
    ds_train = load_split("train", max_samples=N_TRAIN)
    ds_val   = load_split("validation", max_samples=N_VAL)

    # ── 2. Compensar val si COCO no tiene suficientes imágenes ───────────────
    result = compensate_val(ds_train, ds_val)
    if len(result) == 3:
        ds_train, ds_val, ds_extra = result
        has_extra = True
    else:
        ds_train, ds_val = result
        ds_extra  = None
        has_extra = False

    # ── 3. Exportar a formato YOLO ────────────────────────────────────────────
    print(f"\n[INFO] Exportando train ({len(ds_train)} imgs)...")
    exp_train, skip_train = export_to_yolo(ds_train, "train")

    print(f"[INFO] Exportando val ({len(ds_val)} imgs)...")
    exp_val, skip_val = export_to_yolo(ds_val, "val")

    # Exportar el bloque extra (de train) como val adicional
    if has_extra:
        print(f"[INFO] Exportando val extra ({len(ds_extra)} imgs desde train)...")
        exp_extra, _ = export_to_yolo(ds_extra, "val")
        exp_val += exp_extra

    if skip_train or skip_val:
        print(f"[WARN] Omitidas — train: {skip_train}, val: {skip_val}")

    # ── 4. data.yaml ──────────────────────────────────────────────────────────
    write_data_yaml(exp_train, exp_val)

    # ── 5. Limpiar de memoria ─────────────────────────────────────────────────
    fo.load_dataset(f"coco_bottles_train").delete()
    fo.load_dataset(f"coco_bottles_validation").delete()

    print_summary(exp_train, exp_val)


if __name__ == "__main__":
    main()