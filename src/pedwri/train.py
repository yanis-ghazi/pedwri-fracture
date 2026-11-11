import argparse
from pathlib import Path

from pedwri.prepare import write_dataset_yaml


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="prepared dataset directory")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="yolo11s_1024")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from ultralytics import YOLO

    data_yaml = write_dataset_yaml(a.data, Path(a.project) / "dataset.yaml")
    YOLO(a.model).train(
        data=str(data_yaml),
        imgsz=a.imgsz,
        epochs=a.epochs,
        patience=a.patience,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        seed=a.seed,
        deterministic=True,
        cos_lr=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
