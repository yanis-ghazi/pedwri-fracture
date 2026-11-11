import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FILESTEM_RE = re.compile(
    r"^(?P<patient>\d+)_(?P<timestamp>\d+)_(?P<study>\d+)_(?P<region>[A-Z]+)-"
    r"(?P<side>[LR])(?P<projection>\d)_(?P<sex>[A-Z])(?P<age>\d+)"
)
PROJECTIONS = {"1": "ap", "2": "lateral", "3": "oblique"}
DEFAULT_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
]
AGE_BINS = [0, 5, 10, 14, 25]
AGE_LABELS = ["0-4", "5-9", "10-13", "14+"]
TAGS = ["cast", "initial_exam", "diagnosis_uncertain", "metal", "osteopenia"]


def parse_filestem(stem: str) -> dict | None:
    m = FILESTEM_RE.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    return {
        "filestem": stem,
        "patient_id": d["patient"],
        "timestamp": d["timestamp"],
        "study": d["study"],
        "side": d["side"],
        "projection": PROJECTIONS.get(d["projection"], "other"),
        "sex": d["sex"],
        "age": int(d["age"]),
    }


def age_bin(age, ordered: bool = False) -> pd.Series:
    bins = pd.cut(pd.Series(age), bins=AGE_BINS, right=False, labels=AGE_LABELS)
    return bins if ordered else bins.astype(str)


def _truthy(col: pd.Series) -> np.ndarray:
    if col.dtype == bool:
        return col.to_numpy()
    s = col.astype(str).str.strip().str.lower()
    return s.isin({"1", "1.0", "true", "yes", "y", "x"}).to_numpy()


def _stem_column(raw: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in raw.columns}
    if "filestem" in lower:
        return lower["filestem"]
    for c in raw.columns:
        vals = raw[c].astype(str).map(lambda v: Path(v).stem)
        if vals.map(lambda v: FILESTEM_RE.match(v) is not None).mean() > 0.9:
            return c
    raise ValueError("no column with GRAZPEDWRI-DX file stems found in dataset.csv")


def load_metadata(csv_path: Path, exam_key: str = "study") -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    stems = raw[_stem_column(raw)].astype(str).map(lambda v: Path(v).stem)
    parsed = [parse_filestem(s) for s in stems]
    ok = np.array([p is not None for p in parsed])
    if not ok.all():
        warnings.warn(f"{(~ok).sum()} rows with unparseable file stems dropped")
    df = pd.DataFrame([p for p in parsed if p is not None])
    raw = raw.loc[ok].reset_index(drop=True)

    lower = {c.lower(): c for c in raw.columns}
    for tag in TAGS:
        df[tag] = _truthy(raw[lower[tag]]) if tag in lower else False
    ao_col = next((lower[c] for c in lower if c.startswith("ao")), None)
    if ao_col is None:
        df["ao"] = ""
    else:
        ao = raw[ao_col].fillna("").astype(str).str.strip()
        df["ao"] = ao.where(~ao.str.lower().isin({"nan", "none", "0"}), "").to_numpy()

    key = df["study"] if exam_key == "study" else df["timestamp"]
    df["exam_id"] = df["patient_id"] + "_" + key
    return df


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        rows.append((cls, cx, cy, w, h))
    return rows


def load_boxes(label_files: dict[str, Path]) -> pd.DataFrame:
    rows = [(stem, *r) for stem, p in label_files.items() for r in read_yolo_labels(p)]
    return pd.DataFrame(rows, columns=["filestem", "cls", "cx", "cy", "w", "h"])


def read_names(meta_path: Path | None) -> list[str]:
    if meta_path is None:
        return list(DEFAULT_NAMES)
    import yaml

    names = (yaml.safe_load(meta_path.read_text()) or {}).get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names) if names else list(DEFAULT_NAMES)


def discover(root: Path) -> dict:
    csv = next(iter(sorted(root.rglob("dataset.csv"))), None)
    if csv is None:
        raise FileNotFoundError(f"dataset.csv not found under {root}")
    metas = sorted(p for p in root.rglob("meta.yaml") if (p.parent / "labels").is_dir())
    meta = metas[0] if metas else None
    labels_root = meta.parent / "labels" if meta else root
    labels = {p.stem: p for p in labels_root.rglob("*.txt") if FILESTEM_RE.match(p.stem)}
    images = {p.stem: p for p in root.rglob("*.png") if FILESTEM_RE.match(p.stem)}
    return {"csv": csv, "meta": meta, "labels": labels, "images": images}


def read_images(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"filestem": str, "patient_id": str, "exam_id": str,
                                    "study": str, "timestamp": str}, keep_default_na=False)
