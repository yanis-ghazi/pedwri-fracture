import argparse
import json
from pathlib import Path

import pandas as pd

from pedwri.metadata import read_images
from pedwri.prepare import SPLITS, split_counts, write_dataset_yaml
from pedwri.split import (
    assert_no_leak, leaked_patients, patient_table, split_exams, split_images, split_patients,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-split a prepared dataset without copying images")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--unit", choices=["image", "exam", "patient"], default="image")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    info = json.loads((a.data / "info.json").read_text())
    fid = info["fracture_class_id"]
    images = read_images(a.data / "images.csv")
    boxes = pd.read_csv(a.data / "boxes.csv", dtype={"filestem": str})
    source = {s: (sp, a.data.resolve()) for s, sp in zip(images["filestem"], images["split"])}

    if a.unit == "patient":
        pt = patient_table(images, boxes, fid)
        images["split"] = images["patient_id"].map(split_patients(pt, a.val_frac, a.test_frac, a.seed))
        assert_no_leak(images)
    elif a.unit == "exam":
        images["split"] = split_exams(images, boxes, fid, a.val_frac, a.test_frac, a.seed)
    else:
        images["split"] = split_images(images, boxes, fid, a.val_frac, a.test_frac, a.seed)

    for s in SPLITS:
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    for stem, split in zip(images["filestem"], images["split"]):
        old_split, root = source[stem]
        link = a.out / "images" / split / f"{stem}.png"
        if not link.exists():
            link.symlink_to(root / "images" / old_split / f"{stem}.png")
        label = (root / "labels" / old_split / f"{stem}.txt").read_text()
        (a.out / "labels" / split / f"{stem}.txt").write_text(label)

    images.to_csv(a.out / "images.csv", index=False)
    boxes.to_csv(a.out / "boxes.csv", index=False)
    info.update(split_unit=a.unit, seed=a.seed, counts=split_counts(images, boxes, fid))
    (a.out / "info.json").write_text(json.dumps(info, indent=2))
    write_dataset_yaml(a.out, a.out / "dataset.yaml")
    print(f"{leaked_patients(images)} of {images['patient_id'].nunique()} patients span several splits")
    print(pd.DataFrame(info["counts"]).T.to_string())


if __name__ == "__main__":
    main()
