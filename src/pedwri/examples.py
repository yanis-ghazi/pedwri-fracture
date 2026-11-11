import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pedwri.metadata import read_images

GT = (60, 200, 60)
PRED = (0, 140, 255)


def draw(img: np.ndarray, gt: pd.DataFrame, pred: pd.DataFrame, height: int) -> np.ndarray:
    s = height / img.shape[0]
    out = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    for r in gt.itertuples():
        cv2.rectangle(out, (int((r.cx - r.w / 2) * w), int((r.cy - r.h / 2) * h)),
                      (int((r.cx + r.w / 2) * w), int((r.cy + r.h / 2) * h)), GT, 2)
    for r in pred.itertuples():
        p1, p2 = (int(r.x1 * w), int(r.y1 * h)), (int(r.x2 * w), int(r.y2 * h))
        cv2.rectangle(out, p1, p2, PRED, 2)
        cv2.putText(out, f"{r.conf:.2f}", (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED, 1,
                    cv2.LINE_AA)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Random test exams with reference and predicted fracture boxes")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--images", type=Path, help="defaults to <data>/images/test")
    p.add_argument("--op", default="spec_0.90")
    p.add_argument("--n-pos", type=int, default=3)
    p.add_argument("--n-neg", type=int, default=1)
    p.add_argument("--box-score", type=float, default=0.25)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    images_dir = a.images or a.data / "images" / "test"
    out = a.out or a.results / "examples.png"

    fid = json.loads((a.data / "info.json").read_text())["fracture_class_id"]
    t = json.loads((a.results / "operating_points.json").read_text())[a.op]["threshold"]
    exams = pd.read_csv(a.results / "exam_scores_test.csv", dtype={"exam_id": str, "patient_id": str},
                        keep_default_na=False)
    picked = pd.concat([exams[exams["y"] == 1].sample(a.n_pos, random_state=a.seed),
                        exams[exams["y"] == 0].sample(a.n_neg, random_state=a.seed)])
    images = read_images(a.data / "images.csv")
    boxes = pd.read_csv(a.data / "boxes.csv", dtype={"filestem": str})
    preds = pd.read_csv(a.pred_dir / "predictions_test.csv", dtype={"filestem": str})
    boxes, preds = boxes[boxes["cls"] == fid], preds[(preds["cls"] == fid) & (preds["conf"] >= a.box_score)]

    blocks = []
    for e in picked.itertuples():
        stems = images.loc[images["exam_id"] == e.exam_id].sort_values("projection")["filestem"]
        tiles = []
        for stem in stems:
            img = cv2.imread(str(images_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tiles.append(draw(img, boxes[boxes["filestem"] == stem], preds[preds["filestem"] == stem], a.height))
        if not tiles:
            continue
        views = np.hstack(tiles)
        lines = [f"{e.age} y, {'fracture' if e.y else 'no fracture'}",
                 f"exam score {e.score:.2f}, {'flagged' if e.score > t else 'not flagged'}"]
        width = max(views.shape[1], 230)
        caption = np.zeros((46, width, 3), np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(caption, line, (4, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        views = np.pad(views, ((0, 0), (0, width - views.shape[1]), (0, 0)))
        blocks.append(np.vstack([views, caption]))
    if not blocks:
        raise SystemExit(f"no test images found in {images_dir}")
    gap = np.zeros((blocks[0].shape[0], 12, 3), np.uint8)
    cv2.imwrite(str(out), np.hstack([x for b in blocks for x in (b, gap)][:-1]))
    print(f"{len(blocks)} exams written to {out}")


if __name__ == "__main__":
    main()
