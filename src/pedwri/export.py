import argparse
import shutil
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--out", type=Path, default=Path("models"))
    a = p.parse_args(argv)

    from ultralytics import YOLO

    onnx_path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, a.out / "model.onnx")
    print(f"exported to {a.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
