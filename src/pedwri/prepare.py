import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from pedwri.metadata import discover, load_boxes, load_metadata, read_names
from pedwri.preprocess import read_image, standardize
from pedwri.split import assert_no_leak, patient_table, split_patients

SPLITS = ("train", "val", "test")


def _convert(job: tuple[str, str, int]) -> tuple[int, int, int, int]:
    src, dst, max_side = job
    raw = read_image(src)
    out = standardize(raw, max_side)
    cv2.imwrite(dst, out)
    return raw.shape[1], raw.shape[0], out.shape[1], out.shape[0]


def split_counts(images: pd.DataFrame, boxes: pd.DataFrame, fracture_id: int) -> dict:
    fractured = images["filestem"].isin(set(boxes.loc[boxes["cls"] == fracture_id, "filestem"]))
    counts = {}
    for s in SPLITS:
        d = images[images["split"] == s]
        ex = fractured[d.index].groupby(d["exam_id"]).any()
        counts[s] = {
            "patients": int(d["patient_id"].nunique()),
            "exams": int(len(ex)),
            "images": int(len(d)),
            "positive_exams": int(ex.sum()),
        }
    return counts


def write_dataset_yaml(prepared: Path, dest: Path) -> Path:
    info = json.loads((prepared / "info.json").read_text())
    lines = [f"path: {prepared.resolve()}"]
    lines += [f"{s}: images/{s}" for s in SPLITS]
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(info["names"])]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description="Patient-level split and 8-bit conversion of GRAZPEDWRI-DX")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exam-key", choices=["study", "timestamp"], default="study")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-images", action="store_true")
    a = p.parse_args(argv)

    found = discover(a.root)
    names = read_names(found["meta"])
    fracture_id = names.index("fracture")

    images = load_metadata(found["csv"], a.exam_key)
    missing = ~images["filestem"].isin(found["images"].keys())
    if missing.any():
        print(f"warning: {missing.sum()} rows of dataset.csv have no image file, dropped")
    images = images[~missing].reset_index(drop=True)
    labels = {s: found["labels"][s] for s in images["filestem"] if s in found["labels"]}
    boxes = load_boxes(labels)

    pt = patient_table(images, boxes, fracture_id)
    images["split"] = images["patient_id"].map(split_patients(pt, a.val_frac, a.test_frac, a.seed))
    assert_no_leak(images)

    a.out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)

    by_stem = {k: g for k, g in boxes.groupby("filestem")}
    for stem, split in zip(images["filestem"], images["split"]):
        g = by_stem.get(stem)
        text = "" if g is None else "".join(
            f"{int(r.cls)} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}\n" for r in g.itertuples()
        )
        (a.out / "labels" / split / f"{stem}.txt").write_text(text)

    if not a.skip_images:
        jobs = [
            (str(found["images"][s]), str(a.out / "images" / sp / f"{s}.png"), a.max_side)
            for s, sp in zip(images["filestem"], images["split"])
        ]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            sizes = list(ex.map(_convert, jobs, chunksize=32))
        images[["orig_width", "orig_height", "width", "height"]] = pd.DataFrame(sizes)

    images.to_csv(a.out / "images.csv", index=False)
    boxes.to_csv(a.out / "boxes.csv", index=False)

    counts = split_counts(images, boxes, fracture_id)
    info = {
        "names": names,
        "fracture_class_id": fracture_id,
        "split_unit": "patient",
        "seed": a.seed,
        "exam_key": a.exam_key,
        "max_side": a.max_side,
        "counts": counts,
    }
    (a.out / "info.json").write_text(json.dumps(info, indent=2))
    write_dataset_yaml(a.out, a.out / "dataset.yaml")

    n_exams = images["exam_id"].nunique()
    print(f"{len(images)} images, {n_exams} exams, {images['patient_id'].nunique()} patients "
          f"({len(images) / n_exams:.2f} images per exam)")
    print(pd.DataFrame(counts).T.to_string())


if __name__ == "__main__":
    main()
