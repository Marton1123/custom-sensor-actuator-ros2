#!/usr/bin/env python3
"""
01_download_and_prepare.py  (v2 — bottle vs can, 2 clases)
===========================================================
Paso 1 del pipeline: agrega latas (Open Images V7) al dataset de
botellas COCO ya descargado, y re-splitea todo a 80/20.

Estrategia:
    - Bottles : reutiliza data/raw/ existente (COCO, clase 0) — SIN borrar
    - Latas   : descarga ~3000 imgs de Open Images V7  (clase 1)
    - Mezcla  : barajea con seed fijo → ~4800 train / ~1200 val

Estructura de salida (sobreescribe data/raw/ solo al final):
    data/
    └── raw/
        ├── images/  train/ val/
        ├── labels/  train/ val/
        └── data.yaml   ← nc: 2  (0=bottle, 1=can)

Uso:
    conda activate botellas
    python scripts/01_download_and_prepare.py
"""

import random
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# ─── Configuración ────────────────────────────────────────────────────────────
YOLO_ID_BOTTLE = 0
YOLO_ID_CAN    = 1

N_CAN_DOWNLOAD = 3000   # latas a descargar de Open Images
TRAIN_RATIO    = 0.80
SEED           = 42

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
ZOO_DIR  = BASE_DIR / "data" / "_fiftyone_zoo"

# Directorios actuales de bottles (COCO ya descargado)
BOTTLE_IMG_TRAIN = RAW_DIR / "images" / "train"
BOTTLE_IMG_VAL   = RAW_DIR / "images" / "val"
BOTTLE_LBL_TRAIN = RAW_DIR / "labels" / "train"
BOTTLE_LBL_VAL   = RAW_DIR / "labels" / "val"

# Staging temporal para latas (se elimina al final)
STAGE_DIR    = BASE_DIR / "data" / "_staging"
CAN_IMG_DIR  = STAGE_DIR / "images"
CAN_LBL_DIR  = STAGE_DIR / "labels"
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


# ─── 1. Verificar bottles existentes ─────────────────────────────────────────

def check_existing_bottles() -> tuple[list[Path], list[Path]]:
    """
    Lee los pares (imagen, etiqueta) ya existentes de COCO bottles.
    Las etiquetas ya tienen clase 0 — no se modifican.
    """
    print("\n[INFO] Verificando dataset de bottles existente (COCO)...")

    pairs: list[tuple[Path, Path]] = []
    for img_dir, lbl_dir in [
        (BOTTLE_IMG_TRAIN, BOTTLE_LBL_TRAIN),
        (BOTTLE_IMG_VAL,   BOTTLE_LBL_VAL),
    ]:
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                pairs.append((img, lbl))

    assert len(pairs) > 0, (
        f"No se encontraron bottles en {RAW_DIR}.\n"
        "Verifica que data/raw/images/ y data/raw/labels/ existan."
    )

    imgs = [p[0] for p in pairs]
    lbls = [p[1] for p in pairs]
    print(f"[INFO] Bottles encontradas: {len(imgs)} pares imagen/etiqueta (clase 0)")
    return imgs, lbls


# ─── 2. Descargar latas de Open Images V7 ────────────────────────────────────

def download_cans() -> tuple[list[Path], list[Path]]:
    """
    Descarga imágenes de 'Tin can' desde Open Images V7 via FiftyOne.
    Exporta a STAGE_DIR con clase YOLO 1.
    """
    print(f"\n[INFO] Descargando ~{N_CAN_DOWNLOAD} imágenes de latas (Open Images V7)...")

    fo.config.dataset_zoo_dir = str(ZOO_DIR)
    CAN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    CAN_LBL_DIR.mkdir(parents=True, exist_ok=True)

    n_train_dl = int(N_CAN_DOWNLOAD * 0.85)
    n_val_dl   = N_CAN_DOWNLOAD - n_train_dl

    all_can_imgs: list[Path] = []
    all_can_lbls: list[Path] = []

    for split, n_split in [("train", n_train_dl), ("validation", n_val_dl)]:
        ds_name = f"openimages_cans_{split}"

        if fo.dataset_exists(ds_name):
            print(f"[INFO] Dataset '{ds_name}' ya en caché, cargando...")
            ds = fo.load_dataset(ds_name)
        else:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=["Tin can"],
                only_matching=True,
                max_samples=n_split,
                dataset_name=ds_name,
                seed=SEED,
            )
        print(f"[INFO] Open Images '{split}': {len(ds)} imágenes descargadas")

        # Verificar el nombre exacto de la clase en este dataset
        labels_found = ds.distinct("ground_truth.detections.label")
        print(f"[DEBUG] Etiquetas en '{split}': {labels_found[:10]}")

        # Buscar la variante correcta de "tin can" (case-insensitive)
        can_label = next(
            (l for l in labels_found if l.lower() == "tin can"),
            None
        )
        if can_label is None:
            print(f"[WARN] No se encontró 'Tin can' en '{split}'. "
                  f"Etiquetas disponibles: {labels_found[:10]}")
            continue

        if can_label != "Tin can":
            print(f"[INFO] Usando etiqueta '{can_label}' en lugar de 'Tin can'")

        imgs, lbls, skipped = _export_cans_to_yolo(ds, can_label)
        all_can_imgs.extend(imgs)
        all_can_lbls.extend(lbls)

        if skipped:
            print(f"[WARN] Omitidas en '{split}': {skipped}")

    assert len(all_can_imgs) > 0, (
        "No se exportaron latas. Revisa la conexión o el nombre de clase en Open Images."
    )
    print(f"[INFO] Total latas exportadas: {len(all_can_imgs)}")
    return all_can_imgs, all_can_lbls


def _export_cans_to_yolo(
    dataset: fo.Dataset,
    can_label: str,
) -> tuple[list[Path], list[Path], int]:
    """
    Copia imágenes y genera etiquetas YOLO (clase 1) en STAGE_DIR.
    FiftyOne bbox: [x_min_rel, y_min_rel, w_rel, h_rel]
    YOLO bbox:     [x_center_rel, y_center_rel, w_rel, h_rel]
    Retorna (imgs_exportadas, lbls_exportadas, n_omitidas).
    """
    imgs_out: list[Path] = []
    lbls_out: list[Path] = []
    skipped = 0

    for sample in dataset:
        if sample.ground_truth is None:
            skipped += 1
            continue

        can_dets = [
            d for d in sample.ground_truth.detections
            if d.label == can_label
        ]
        if not can_dets:
            skipped += 1
            continue

        src = Path(sample.filepath)
        # Prefijo "can_" para evitar colisiones de nombre con bottles
        dst_name = "can_" + src.name
        dst_img  = CAN_IMG_DIR / dst_name

        if not dst_img.exists():
            shutil.copy2(src, dst_img)

        lines = []
        for det in can_dets:
            x, y, w, h = det.bounding_box
            x_center = max(0.0, min(1.0, x + w / 2.0))
            y_center = max(0.0, min(1.0, y + h / 2.0))
            w        = max(0.001, min(1.0, w))
            h        = max(0.001, min(1.0, h))
            lines.append(
                f"{YOLO_ID_CAN} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        lbl_file = CAN_LBL_DIR / Path(dst_name).with_suffix(".txt").name
        lbl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        imgs_out.append(dst_img)
        lbls_out.append(lbl_file)

    return imgs_out, lbls_out, skipped


# ─── 3. Mezclar y re-splitear ─────────────────────────────────────────────────

def merge_and_split(
    bottle_imgs: list[Path], bottle_lbls: list[Path],
    can_imgs:    list[Path], can_lbls:    list[Path],
) -> tuple[int, int]:
    """
    Mezcla bottles y latas, barajea y re-splitea 80/20.
    Sobreescribe data/raw/ SOLO al final, cuando todos los archivos
    están listos en staging, para no perder el dataset original si
    el script falla a mitad.
    """
    print("\n[INFO] Mezclando y re-spliteando dataset...")

    all_imgs = bottle_imgs + can_imgs
    all_lbls = bottle_lbls + can_lbls

    combined = list(zip(all_imgs, all_lbls))
    assert len(combined) > 0, "No hay imágenes para mezclar. Verifica los pasos anteriores."

    random.shuffle(combined)
    all_imgs_s, all_lbls_s = zip(*combined)

    n_total = len(all_imgs_s)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = n_total - n_train
    print(f"[INFO] Total: {n_total} | Train: {n_train} | Val: {n_val}")

    # Preparar en staging antes de tocar data/raw/
    out_stage = STAGE_DIR / "final"
    splits = {
        "train": (list(all_imgs_s[:n_train]), list(all_lbls_s[:n_train])),
        "val":   (list(all_imgs_s[n_train:]), list(all_lbls_s[n_train:])),
    }

    for split_name, (imgs, lbls) in splits.items():
        img_out = out_stage / "images" / split_name
        lbl_out = out_stage / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in zip(imgs, lbls):
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)

    print("[INFO] Staging listo. Reemplazando data/raw/...")

    # Solo aquí se toca data/raw/ — staging completado con éxito
    for split_name in ["train", "val"]:
        for kind in ["images", "labels"]:
            dst = RAW_DIR / kind / split_name
            src = out_stage / kind / split_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    _print_class_balance("train", RAW_DIR / "labels" / "train")
    _print_class_balance("val",   RAW_DIR / "labels" / "val")

    return n_train, n_val


def _print_class_balance(split: str, lbl_dir: Path) -> None:
    counts: dict[int, int] = {}
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    b = counts.get(0, 0)
    c = counts.get(1, 0)
    print(
        f"[INFO] Balance {split}: "
        f"bottle={b} ({b/max(total,1)*100:.1f}%) | "
        f"can={c} ({c/max(total,1)*100:.1f}%)"
    )


# ─── 4. data.yaml ─────────────────────────────────────────────────────────────

def write_data_yaml(n_train: int, n_val: int) -> None:
    yaml_text = f"""\
# ── Dataset: Bottle vs Can (2 clases) ────────────────────────
# Generado por 01_download_and_prepare.py  (v2)
# train: {n_train} imágenes  |  val: {n_val} imágenes
# Fuentes: COCO 2017 (bottles) + Open Images V7 (cans)

path: {RAW_DIR.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: bottle
  1: can
"""
    out = RAW_DIR / "data.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    print(f"[INFO] data.yaml escrito en: {out}")


# ─── 5. Limpieza ──────────────────────────────────────────────────────────────

def cleanup(delete_fo_datasets: bool = True) -> None:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
        print(f"[INFO] Staging eliminado: {STAGE_DIR}")

    if delete_fo_datasets:
        for ds_name in ["openimages_cans_train", "openimages_cans_validation"]:
            if fo.dataset_exists(ds_name):
                fo.load_dataset(ds_name).delete()
                print(f"[INFO] Dataset FiftyOne '{ds_name}' eliminado de memoria")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("  Pipeline Paso 1 v2 — Bottle vs Can (2 clases)")
    print("=" * 58)

    # 1. Leer bottles ya existentes (COCO, no se redescargan)
    bottle_imgs, bottle_lbls = check_existing_bottles()

    # 2. Descargar latas desde Open Images V7
    can_imgs, can_lbls = download_cans()

    # 3. Mezclar, barajear y re-splitear — data/raw/ se toca solo al final
    n_train, n_val = merge_and_split(
        bottle_imgs, bottle_lbls,
        can_imgs,    can_lbls,
    )

    # 4. Escribir data.yaml con nc=2
    write_data_yaml(n_train, n_val)

    # 5. Limpiar staging y datasets FiftyOne temporales
    cleanup()

    print("\n" + "=" * 58)
    print("  ✅  Pipeline Paso 1 v2 completado")
    print("=" * 58)
    print(f"  Train : {n_train:>5} imágenes  →  data/raw/images/train/")
    print(f"  Val   : {n_val:>5} imágenes  →  data/raw/images/val/")
    print(f"  Clases: 0 = bottle  |  1 = can")
    print(f"  YAML  : data/raw/data.yaml")
    print("=" * 58)
    print("\n  Siguiente paso:")
    print("  python scripts/02_train.py")


if __name__ == "__main__":
    main()