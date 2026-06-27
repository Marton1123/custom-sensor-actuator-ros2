#!/usr/bin/env python3
"""Convierte el dataset YOLO del proyecto al ZIP Pascal VOC de maix_train."""

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image


CLASSES = ("bottle", "can")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def yolo_to_voc(label_path, image_path, destination):
    with Image.open(image_path) as image:
        width, height = image.size
        depth = len(image.getbands())

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_path.name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    if label_path.exists():
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_number}: etiqueta YOLO inválida")
            class_id = int(parts[0])
            if not 0 <= class_id < len(CLASSES):
                raise ValueError(f"{label_path}:{line_number}: clase {class_id} desconocida")
            xc, yc, bw, bh = map(float, parts[1:])
            xmin = max(0, round((xc - bw / 2) * width))
            ymin = max(0, round((yc - bh / 2) * height))
            xmax = min(width - 1, round((xc + bw / 2) * width))
            ymax = min(height - 1, round((yc + bh / 2) * height))
            if xmax <= xmin or ymax <= ymin:
                continue

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = CLASSES[class_id]
            ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(xmin)
            ET.SubElement(box, "ymin").text = str(ymin)
            ET.SubElement(box, "xmax").text = str(xmax)
            ET.SubElement(box, "ymax").text = str(ymax)

    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def convert(dataset, output):
    staging = output.parent / f".{output.stem}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    total = 0
    try:
        for split in ("train", "val"):
            image_dir = dataset / "images" / split
            label_dir = dataset / "labels" / split
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Falta images/{split} o labels/{split}")
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # El prefijo evita colisiones entre train y val.
                stem = f"{split}_{image_path.stem}"
                copied_image = staging / f"{stem}{image_path.suffix.lower()}"
                shutil.copy2(image_path, copied_image)
                yolo_to_voc(
                    label_dir / f"{image_path.stem}.txt",
                    image_path,
                    staging / f"{stem}.xml",
                )
                total += 1

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.iterdir()):
                archive.write(path, path.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset K210 preparado: {output} ({total} imágenes)")


def main():
    project_ia = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_ia / "data" / "raw",
        help="Directorio con images/{train,val} y labels/{train,val}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_ia / "data" / "bottle_can_k210_voc.zip",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert(args.dataset.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
