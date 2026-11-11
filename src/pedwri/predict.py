import argparse
import json
from pathlib import Path

import pandas as pd

from pedwri.metrics import PRED_COLUMNS
from pedwri.prepare import write_dataset_yaml


def _map_summary(metrics, names: list[str]) -> dict:
    out = {k: float(v) for k, v in metrics.results_dict.items()}
    box = metrics.box
    try:
        out["per_class"] = {
            names[int(c)]: {"ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i])}
            for i, c in enumerate(box.ap_class_index)
        }
    except AttributeError:
        pass
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", type=Path, required=True, help="prepared dataset directory")
    p.add_argument("--out", type=Path, default=Path("predictions"))
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="0")
    a = p.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = json.loads((a.data / "info.json").read_text())["names"]
    images = pd.read_csv(a.data / "images.csv", dtype={"filestem": str})
    data_yaml = write_dataset_yaml(a.data, a.out / "dataset.yaml")
    a.out.mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        stems = images.loc[images["split"] == split, "filestem"].tolist()
        paths = [str(a.data / "images" / split / f"{s}.png") for s in stems]
        rows = []
        for i in range(0, len(paths), a.batch):
            results = model.predict(
                paths[i:i + a.batch], imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                max_det=100, device=a.device, verbose=False,
            )
            for r in results:
                stem = Path(r.path).stem
                b = r.boxes
                for xyxy, conf, cls in zip(b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                    rows.append((stem, int(cls), float(conf), *map(float, xyxy)))
        pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(a.out / f"predictions_{split}.csv", index=False)

        metrics = model.val(
            data=str(data_yaml), split=split, imgsz=a.imgsz, batch=a.batch,
            conf=a.conf, iou=a.iou, device=a.device, plots=False, verbose=False,
        )
        (a.out / f"map_{split}.json").write_text(json.dumps(_map_summary(metrics, names), indent=2))
        print(f"{split}: {len(rows)} boxes on {len(paths)} images")


if __name__ == "__main__":
    main()
