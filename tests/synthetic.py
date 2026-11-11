from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from pedwri.metadata import DEFAULT_NAMES

AO_CHOICES = ["23r-M/2.1", "23r-M/3.1; 23u-E/7", "23r-E/2.1", "23r-M/3.1; 23u-M/2.1", "23-M/3.1"]


def make_dataset(root: Path, n_patients: int = 120, seed: int = 0, size=(180, 140)) -> Path:
    rng = np.random.default_rng(seed)
    yolo = root / "folder_structure" / "yolov5"
    (yolo / "labels").mkdir(parents=True)
    images = root / "images"
    images.mkdir(parents=True)
    (yolo / "meta.yaml").write_text(yaml.safe_dump({"nc": 9, "names": DEFAULT_NAMES}))

    h, w = size
    rows = []
    for p in range(n_patients):
        sex = rng.choice(["M", "F"])
        age = int(rng.integers(1, 19))
        fractured = rng.random() < 0.65
        for study in range(1, int(rng.integers(1, 4)) + 1):
            ts = f"{int(rng.integers(10**8, 10**9)):010d}"
            initial = study == 1
            visible = fractured and (initial or rng.random() < 0.5)
            code = rng.choice(AO_CHOICES) if fractured and initial else ""
            for proj in (1, 2):
                stem = f"{p:04d}_{ts}_{study:02d}_WRI-L{proj}_{sex}{age:03d}"
                img = (np.linspace(8000, 40000, w)[None, :] * np.ones((h, 1))).astype(np.float32)
                img += rng.normal(0, 800, (h, w))
                lines = ["8 0.08 0.06 0.06 0.05"]
                if visible:
                    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
                    bw, bh = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25)
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    img[y1:y2, x1:x2] += 15000
                    lines.append(f"3 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                cv2.imwrite(str(images / f"{stem}.png"), np.clip(img, 0, 65535).astype(np.uint16))
                (yolo / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                rows.append({
                    "filestem": stem,
                    "ao_classification": code,
                    "cast": int(not initial),
                    "initial_exam": int(initial),
                    "diagnosis_uncertain": int(rng.random() < 0.03),
                    "metal": 0,
                    "osteopenia": int(rng.random() < 0.1),
                    "fracture_visible": int(visible),
                })
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    return root


def fake_predictions(prepared: Path, split: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    images = pd.read_csv(prepared / "images.csv", dtype={"filestem": str})
    boxes = pd.read_csv(prepared / "boxes.csv", dtype={"filestem": str})
    stems = images.loc[images["split"] == split, "filestem"]
    gt = boxes[boxes["filestem"].isin(stems)]
    rows = []
    for r in gt.itertuples():
        if r.cls != 3 or rng.random() < 0.15:
            continue
        j = rng.normal(0, 0.01, 4)
        rows.append((r.filestem, 3, float(rng.beta(5, 2)), r.cx - r.w / 2 + j[0], r.cy - r.h / 2 + j[1],
                     r.cx + r.w / 2 + j[2], r.cy + r.h / 2 + j[3]))
    for s in stems:
        for _ in range(int(rng.integers(0, 3))):
            x, y = rng.uniform(0.1, 0.8, 2)
            rows.append((s, 3, float(rng.beta(1.5, 5)), x, y, x + 0.1, y + 0.1))
    return pd.DataFrame(rows, columns=["filestem", "cls", "conf", "x1", "y1", "x2", "y2"])
