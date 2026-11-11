import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedwri import ao
from pedwri.metadata import age_bin
from pedwri.metrics import wilson


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def rate_table(df: pd.DataFrame, by: str, flag: str, min_n: int = 1) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[flag].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    g = g[g["n"] >= min_n]
    g["rate"] = g["k"] / g["n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    g.index.name = by
    return g.reset_index()


def explode(df: pd.DataFrame, fn, col: str, empty: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = [sorted(fn(v)) or [empty] for v in out["ao"].fillna("")]
    return out.explode(col)


def size_bins(lesions: pd.DataFrame) -> pd.Series:
    side = np.sqrt(lesions["w"] * lesions["h"])
    return pd.qcut(side, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])


def plot_rates(tables: dict[str, pd.DataFrame], xlabel: str, path: Path) -> None:
    n = len(tables)
    heights = [max(1.2, 0.32 * len(t) + 0.6) for t in tables.values()]
    fig, axes = plt.subplots(n, 1, figsize=(6, sum(heights)), gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    xmax = min(1.0, max(t["ci_high"].max() for t in tables.values()) + 0.05)
    for ax, (name, t) in zip(axes, tables.items()):
        t = t.iloc[::-1]
        y = np.arange(len(t))
        err = np.vstack([t["rate"] - t["ci_low"], t["ci_high"] - t["rate"]])
        ax.barh(y, t["rate"], xerr=err, color="0.55", ecolor="0.2", capsize=2, height=0.6)
        ax.set_yticks(y, [f"{v} (n={n_})" for v, n_ in zip(t.iloc[:, 0], t["n"])], fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gallery(lesions: pd.DataFrame, images_dir: Path, path: Path, k: int = 16, tile: int = 220) -> None:
    rows = []
    for r in lesions.head(k).itertuples():
        img = cv2.imread(str(images_dir / f"{r.filestem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        x1, y1, x2, y2 = r.x1 * w, r.y1 * h, r.x2 * w, r.y2 * h
        side = max(x2 - x1, y2 - y1) * 2.5 + 40
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        c, d = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        crop = cv2.cvtColor(img[b:d, a:c], cv2.COLOR_GRAY2BGR)
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s)
        cv2.rectangle(crop, (int((x1 - a) * s), int((y1 - b) * s)), (int((x2 - a) * s), int((y2 - b) * s)),
                      (60, 200, 60), 1)
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        label = f"{r.age}y {r.ao.split(';')[0] if r.ao else ''}"
        cv2.putText(canvas, label, (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(canvas)
    if not rows:
        return
    while len(rows) % 4:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + 4]) for i in range(0, len(rows), 4)])
    cv2.imwrite(str(path), grid)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--op", default="spec_0.90")
    p.add_argument("--images", type=Path, help="prepared test images, for the gallery of missed fractures")
    p.add_argument("--min-n", type=int, default=10)
    a = p.parse_args(argv)
    out = a.results / "errors"
    out.mkdir(parents=True, exist_ok=True)

    t = json.loads((a.results / "operating_points.json").read_text())[a.op]["threshold"]
    exams = pd.read_csv(a.results / "exam_scores_test.csv", keep_default_na=False, dtype={"exam_id": str})
    lesions = pd.read_csv(a.results / "lesions_test.csv", keep_default_na=False, dtype={"filestem": str})

    pos = exams[exams["y"] == 1].copy()
    pos["missed"] = pos["score"] <= t
    pos["age_group"] = age_bin(pos["age"], ordered=True).values
    pos["exam_type"] = np.where(pos["initial_exam"], "initial", "follow-up")
    pos["cast_"] = np.where(pos["cast"], "cast", "no cast")

    exam_tables = {
        "age": rate_table(pos, "age_group", "missed"),
        "fracture pattern": rate_table(explode(pos, ao.patterns, "pattern", "no AO code"), "pattern",
                                       "missed", a.min_n),
        "AO bone/segment/type": rate_table(explode(pos, ao.detailed, "ao_group", "no AO code"), "ao_group",
                                           "missed", a.min_n),
        "exam": rate_table(pos, "exam_type", "missed"),
        "cast": rate_table(pos, "cast_", "missed"),
        "sex": rate_table(pos, "sex", "missed"),
    }

    lesions["missed"] = lesions["hit_score"] <= t
    lesions["size"] = size_bins(lesions)
    lesions["age_group"] = age_bin(lesions["age"], ordered=True).values
    lesion_tables = {
        "box size": rate_table(lesions, "size", "missed"),
        "projection": rate_table(lesions, "projection", "missed"),
        "age": rate_table(lesions, "age_group", "missed"),
        "fracture pattern": rate_table(explode(lesions, ao.patterns, "pattern", "no AO code"), "pattern",
                                       "missed", a.min_n),
    }

    neg = exams[exams["y"] == 0].copy()
    neg["false_positive"] = neg["score"] > t
    neg["ao_without_box"] = np.where(neg["ao"].astype(str).str.len() > 0, "AO code, no box", "no AO code")
    neg["cast_"] = np.where(neg["cast"], "cast", "no cast")
    neg["exam_type"] = np.where(neg["initial_exam"], "initial", "follow-up")
    fp_tables = {
        "label": rate_table(neg, "ao_without_box", "false_positive"),
        "cast": rate_table(neg, "cast_", "false_positive"),
        "exam": rate_table(neg, "exam_type", "false_positive"),
    }

    lines = [f"Operating point {a.op} (threshold {t:.3f}).", ""]
    sections = [
        ("missed_exams", "Missed positive exams", exam_tables),
        ("missed_lesions", "Missed fracture boxes", lesion_tables),
        ("false_positives", "False positive rate among negative exams", fp_tables),
    ]
    for prefix, title, tables in sections:
        lines.append(f"{title}\n")
        for name, tab in tables.items():
            tab.to_csv(out / f"{prefix}_by_{slug(name)}.csv", index=False)
            lines.append(f"by {name}\n")
            lines.append("| group | n | rate (95% CI) |")
            lines.append("|---|---|---|")
            for r in tab.itertuples(index=False):
                lines.append(f"| {r[0]} | {r.n} | {r.rate:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f}) |")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines))

    plot_rates(exam_tables, "Missed positive exams (1 - sensitivity)", out / "missed_exams.png")
    plot_rates(lesion_tables, "Missed fracture boxes", out / "missed_lesions.png")
    plot_rates(fp_tables, "False positive rate", out / "false_positives.png")

    missed = lesions[lesions["missed"]].assign(area=lambda d: d["w"] * d["h"]).sort_values("area", ascending=False)
    missed.to_csv(out / "missed_lesions.csv", index=False)
    if a.images is not None:
        gallery(missed, a.images, out / "missed_largest.png")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
