#!/usr/bin/env python3
"""
dataset_pipeline.py
===================
Pipeline local (sin ROS) para Windows/Linux:
1) Capturar video desde camara
2) Extraer frames con botella usando YOLOv8n COCO (clase bottle=39)
3) Subir imagenes filtradas a Roboflow

Ejemplo rapido (Windows):
    python scripts/dataset_pipeline.py run ^
      --duration 180 ^
      --out-root .\dataset_work ^
      --upload ^
      --roboflow-api-key TU_KEY ^
      --roboflow-workspace TU_WORKSPACE ^
      --roboflow-project TU_PROYECTO
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _open_camera(cam_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def capture_video(
    out_video: Path,
    cam_index: int,
    width: int,
    height: int,
    fps: int,
    duration_s: int,
    show_preview: bool,
) -> Path:
    cap = _open_camera(cam_index, width, height, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara (index={cam_index}).")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el video de salida: {out_video}")

    print(f"[capture] Grabando {duration_s}s en: {out_video}")
    print("[capture] Presiona 'q' para terminar antes.")
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        if show_preview:
            cv2.imshow("Captura Dataset", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if time.time() - t0 >= duration_s:
            break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    return out_video


def filter_frames_with_bottle(
    in_video: Path,
    out_frames_dir: Path,
    sample_fps: float,
    conf_thres: float,
    imgsz: int,
    model_name: str,
    class_id: int,
    max_frames: int,
) -> tuple[int, int, Path]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Instala ultralytics: pip install ultralytics") from exc

    _ensure_dir(out_frames_dir)
    manifest = out_frames_dir / "manifest.csv"

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {in_video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps))) if sample_fps > 0 else 1

    kept = 0
    seen = 0
    frame_idx = 0
    stem = in_video.stem

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "video", "frame_idx", "best_conf", "area_px"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            seen += 1
            results = model(
                frame,
                verbose=False,
                imgsz=imgsz,
                conf=conf_thres,
                classes=[class_id],
            )
            boxes = results[0].boxes if results and len(results) else None
            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            best_conf = 0.0
            best_area = 0.0
            for box in boxes:
                conf = float(box.conf[0])
                w = float(box.xywh[0][2])
                h = float(box.xywh[0][3])
                area = w * h
                if conf > best_conf:
                    best_conf = conf
                    best_area = area

            out_name = f"{stem}_f{frame_idx:07d}_c{best_conf:.2f}.jpg"
            out_path = out_frames_dir / out_name
            cv2.imwrite(str(out_path), frame)
            writer.writerow([out_name, in_video.name, frame_idx, f"{best_conf:.4f}", f"{best_area:.2f}"])
            kept += 1

            if max_frames > 0 and kept >= max_frames:
                break

            frame_idx += 1

    cap.release()
    return seen, kept, manifest


def _split_items(items: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Ratios invalidos: usa train>0, val>=0 y train+val<1.")
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return {"train": train, "valid": val, "test": test}


def upload_to_roboflow(
    frames_dir: Path,
    api_key: str,
    workspace: str,
    project_slug: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    batch_name: str | None,
) -> None:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Instala roboflow: pip install roboflow") from exc

    images = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg")) + sorted(frames_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No hay imagenes para subir en: {frames_dir}")

    splits = _split_items(images, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    batch = batch_name or f"dataset_{_timestamp()}"

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)

    total = 0
    failed = 0
    for split_name, split_images in splits.items():
        print(f"[upload] split={split_name} | imagenes={len(split_images)}")
        for img in split_images:
            total += 1
            try:
                project.upload(
                    image_path=str(img),
                    split=split_name,
                    batch_name=batch,
                    num_retry_uploads=3,
                )
            except Exception:
                failed += 1
    print(f"[upload] completado | total={total} | failed={failed} | batch={batch}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline local de dataset para botellas (sin ROS).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Captura video, filtra frames con botella y opcionalmente sube a Roboflow.")
    run.add_argument("--out-root", type=Path, default=Path("dataset_work"))
    run.add_argument("--duration", type=int, default=180, help="Duracion de grabacion en segundos.")
    run.add_argument("--cam-index", type=int, default=0)
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--fps", type=int, default=30)
    run.add_argument("--no-preview", action="store_true")
    run.add_argument("--sample-fps", type=float, default=3.0, help="FPS a muestrear para extraer candidatos.")
    run.add_argument("--conf-thres", type=float, default=0.35, help="Confianza minima para prefiltrado bottle COCO.")
    run.add_argument("--imgsz", type=int, default=640)
    run.add_argument("--prefilter-model", type=str, default="yolov8n.pt")
    run.add_argument("--class-id", type=int, default=39, help="Clase bottle en COCO.")
    run.add_argument("--max-frames", type=int, default=0, help="0=sin limite.")
    run.add_argument("--upload", action="store_true", help="Subir frames filtrados a Roboflow.")
    run.add_argument("--roboflow-api-key", type=str, default="")
    run.add_argument("--roboflow-workspace", type=str, default="")
    run.add_argument("--roboflow-project", type=str, default="")
    run.add_argument("--train-ratio", type=float, default=0.8)
    run.add_argument("--val-ratio", type=float, default=0.1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--batch-name", type=str, default="")

    return p


def run_pipeline(args: argparse.Namespace) -> None:
    out_root = _ensure_dir(args.out_root)
    videos_dir = _ensure_dir(out_root / "videos")
    frames_dir = _ensure_dir(out_root / "frames_filtered")

    video_path = videos_dir / f"dataset_{_timestamp()}.mp4"
    capture_video(
        out_video=video_path,
        cam_index=args.cam_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_s=args.duration,
        show_preview=not args.no_preview,
    )

    seen, kept, manifest = filter_frames_with_bottle(
        in_video=video_path,
        out_frames_dir=frames_dir,
        sample_fps=args.sample_fps,
        conf_thres=args.conf_thres,
        imgsz=args.imgsz,
        model_name=args.prefilter_model,
        class_id=args.class_id,
        max_frames=args.max_frames,
    )
    print(f"[filter] frames evaluados={seen} | frames guardados={kept}")
    print(f"[filter] manifest: {manifest}")

    if not args.upload:
        return

    if not (args.roboflow_api_key and args.roboflow_workspace and args.roboflow_project):
        raise RuntimeError("Para --upload debes proveer API key, workspace y project.")

    upload_to_roboflow(
        frames_dir=frames_dir,
        api_key=args.roboflow_api_key,
        workspace=args.roboflow_workspace,
        project_slug=args.roboflow_project,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_name=args.batch_name.strip() or None,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.cmd == "run":
        run_pipeline(args)


if __name__ == "__main__":
    main()
