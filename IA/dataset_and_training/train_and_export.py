#!/usr/bin/env python3
"""
train_and_export.py
===================
Script de entrenamiento YOLOv8n y exportación a NCNN INT8.

Ejecutar en el equipo de desarrollo local (Ryzen 5 4600H + GTX 1650).
NO ejecutar en la Raspberry Pi.

Prerrequisitos (en el entorno local con GPU):
    pip install ultralytics roboflow

Uso:
    python train_and_export.py \
        --roboflow-api-key TU_API_KEY \
        --roboflow-workspace TU_WORKSPACE \
        --roboflow-project TU_PROYECTO \
        --roboflow-version 1 \
        --epochs 100 \
        --imgsz 640 \
        --batch 16

Salida:
    - Modelo entrenado en runs/detect/botella_yolov8n/weights/best.pt
    - Modelo exportado NCNN INT8 en runs/detect/botella_yolov8n/weights/best_ncnn_model/
    
Copiar la carpeta `best_ncnn_model/` a la Raspberry Pi 5:
    scp -r runs/detect/botella_yolov8n/weights/best_ncnn_model/ pi@<IP_RPI>:~/modelos/
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena YOLOv8n con dataset de Roboflow y exporta a NCNN INT8."
    )
    # ── Roboflow ──────────────────────────────────────────────────────
    parser.add_argument(
        "--roboflow-api-key", type=str, required=True,
        help="API key de tu cuenta de Roboflow.",
    )
    parser.add_argument(
        "--roboflow-workspace", type=str, required=True,
        help="Nombre del workspace en Roboflow.",
    )
    parser.add_argument(
        "--roboflow-project", type=str, required=True,
        help="Nombre del proyecto (slug) en Roboflow.",
    )
    parser.add_argument(
        "--roboflow-version", type=int, default=1,
        help="Versión del dataset en Roboflow [default: 1].",
    )

    # ── Entrenamiento ─────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Modelo base de Ultralytics [default: yolov8n.pt].",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número de épocas de entrenamiento [default: 100].",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamaño de imagen de entrada en píxeles [default: 640].",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Tamaño de batch [default: 16]. Ajustar según VRAM de la GPU.",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Dispositivo CUDA ('0', '1', 'cpu') [default: '0' = primera GPU].",
    )
    parser.add_argument(
        "--name", type=str, default="botella_yolov8n",
        help="Nombre del experimento / directorio de salida [default: botella_yolov8n].",
    )

    # ── Exportación ───────────────────────────────────────────────────
    parser.add_argument(
        "--export-imgsz", type=int, default=640,
        help="Tamaño de imagen para el modelo exportado [default: 640].",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Solo entrenar, no exportar a NCNN.",
    )

    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> str:
    """
    Descarga el dataset desde Roboflow en formato YOLOv8.
    Retorna la ruta al directorio del dataset descargado.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: Instala roboflow → pip install roboflow")
        sys.exit(1)

    print("=" * 60)
    print("PASO 1: Descargando dataset desde Roboflow...")
    print("=" * 60)

    rf = Roboflow(api_key=args.roboflow_api_key)
    project = rf.workspace(args.roboflow_workspace).project(args.roboflow_project)
    version = project.version(args.roboflow_version)

    # Descarga en formato YOLOv8
    dataset = version.download("yolov8")

    dataset_path = dataset.location
    print(f"Dataset descargado en: {dataset_path}")
    return dataset_path


def train_model(args: argparse.Namespace, dataset_path: str) -> tuple[str, str]:
    """
    Entrena YOLOv8n sobre el dataset descargado.
    Retorna la ruta al modelo best.pt y al data.yaml del dataset.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Instala ultralytics → pip install ultralytics")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PASO 2: Entrenando YOLOv8n...")
    print("=" * 60)

    # El archivo data.yaml se genera automáticamente por Roboflow
    data_yaml = str(Path(dataset_path) / "data.yaml")

    model = YOLO(args.model)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,           # early stopping si val loss no mejora en 20 épocas
        save=True,
        save_period=10,        # checkpoint cada 10 épocas
        plots=True,            # genera gráficas de métricas
        verbose=True,
        workers=4,
        exist_ok=True,         # sobreescribe si ya existe el directorio
    )

    # Ruta al mejor modelo
    best_pt = str(Path(results.save_dir) / "weights" / "best.pt")
    print(f"\nModelo entrenado guardado en: {best_pt}")
    return best_pt, data_yaml


def export_ncnn_int8(args: argparse.Namespace, best_pt: str, data_yaml: str) -> str:
    """
    Exporta el modelo entrenado a formato NCNN con cuantización INT8.
    Retorna la ruta a la carpeta del modelo NCNN.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("PASO 3: Exportando a NCNN INT8...")
    print("=" * 60)

    model = YOLO(best_pt)

    # Exportar a NCNN con cuantización INT8
    # Ultralytics genera una carpeta *_ncnn_model/ junto al .pt
    ncnn_path = model.export(
        format="ncnn",
        imgsz=args.export_imgsz,
        half=False,        # no FP16, usamos INT8
        int8=True,         # cuantización INT8 para máximo rendimiento en CPU
        data=data_yaml,    # calibración INT8 con tu dataset real
    )

    print(f"\nModelo NCNN INT8 exportado en: {ncnn_path}")
    print("\n" + "=" * 60)
    print("INSTRUCCIONES PARA DESPLIEGUE EN RASPBERRY PI 5:")
    print("=" * 60)
    print(f"  scp -r {ncnn_path} pi@<IP_RPI>:~/modelos/")
    print("  # Luego en params.yaml, configurar:")
    print("  #   modelo_path: '/home/pi/modelos/best_ncnn_model'")
    print("=" * 60)

    return ncnn_path


def main() -> None:
    args = parse_args()

    # 1. Descargar dataset
    dataset_path = download_dataset(args)

    # 2. Entrenar
    best_pt, data_yaml = train_model(args, dataset_path)

    # 3. Exportar (opcional)
    if not args.skip_export:
        export_ncnn_int8(args, best_pt, data_yaml)
    else:
        print("\nExportación NCNN omitida (--skip-export).")

    print("\n✅ Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
