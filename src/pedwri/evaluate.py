import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from pedwri.metadata import read_images
from pedwri.metrics import (
    auc, binary_metrics, bootstrap_ci, exam_table, froc, image_table, match_lesions,
    sensitivity_at_fppi, threshold_at_specificity,
)

FPPI_POINTS = [0.25, 0.5, 1.0, 2.0]
SUBGROUPS = {
    "all": lambda e: e,
    "initial_exam": lambda e: e[e["initial_exam"]],
    "follow_up": lambda e: e[~e["initial_exam"]],
    "certain_labels": lambda e: e[~e["diagnosis_uncertain"]],
    "two_views": lambda e: e[e["n_images"] >= 2],
}


def exam_report(exams: pd.DataFrame, ops: dict, n_boot: int, seed: int) -> dict:
    y = exams["y"].to_numpy()
    s = exams["score"].to_numpy()

    def stats(idx):
        out = {"auc": auc(y[idx], s[idx])}
        for name, op in ops.items():
            m = binary_metrics(y[idx], s[idx], op["threshold"])
            for k in ("sensitivity", "specificity", "ppv", "npv"):
                out[f"{name}/{k}"] = m[k]
        return out

    ci = bootstrap_ci(exams["patient_id"].to_numpy(), stats, n_boot=n_boot, seed=seed)
    report = {
        "n_exams": int(len(exams)),
        "n_positive": int(y.sum()),
        "n_patients": int(exams["patient_id"].nunique()),
        "auc": ci["auc"],
        "operating_points": {},
    }
    for name, op in ops.items():
        counts = binary_metrics(y, s, op["threshold"])
        report["operating_points"][name] = {
            "threshold": op["threshold"],
            **{k: ci[f"{name}/{k}"] for k in ("sensitivity", "specificity", "ppv", "npv")},
            **{k: counts[k] for k in ("tp", "fn", "fp", "tn")},
        }
    return report


def fmt(d: dict) -> str:
    if not np.isfinite(d["value"]):
        return "n/a"
    return f"{d['value']:.3f} ({d['ci_low']:.3f}-{d['ci_high']:.3f})"


def markdown(report: dict, map_test: dict | None) -> str:
    lines = []
    main = report["exam_level"]["all"]
    lines.append(f"Test set: {main['n_exams']} exams from {main['n_patients']} patients, "
                 f"{main['n_positive']} with at least one fracture box.\n")
    lines.append("| Subgroup | Exams | AUC |")
    lines.append("|---|---|---|")
    for name, r in report["exam_level"].items():
        lines.append(f"| {name} | {r['n_exams']} | {fmt(r['auc'])} |")
    lines.append("")
    if "image_level" in report:
        il = report["image_level"]
        lines.append(f"Image level: AUC {fmt(il['auc'])} on {il['n_images']} images, "
                     f"{il['n_positive']} with a fracture box.\n")
    lines.append("| Operating point (val) | Threshold | Sensitivity | Specificity | PPV | NPV |")
    lines.append("|---|---|---|---|---|---|")
    for name, op in main["operating_points"].items():
        lines.append(f"| {name} | {op['threshold']:.3f} | {fmt(op['sensitivity'])} | "
                     f"{fmt(op['specificity'])} | {fmt(op['ppv'])} | {fmt(op['npv'])} |")
    lines.append("")
    lines.append("| Specificity on the test ROC | Sensitivity |")
    lines.append("|---|---|")
    for k, v in report["test_curve_sensitivity_at_specificity"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    les = report["lesion_level"]
    lines.append(f"Lesion level ({les['n_lesions']} fracture boxes, IoU >= {les['iou']}):\n")
    lines.append("| FP per image | Sensitivity |")
    lines.append("|---|---|")
    for k, v in les["sensitivity_at_fppi"].items():
        lines.append(f"| {k} | {v:.3f} |")
    if map_test:
        lines.append("")
        frac = map_test.get("per_class", {}).get("fracture", {})
        lines.append(f"mAP50 (all classes) {map_test.get('metrics/mAP50(B)', float('nan')):.3f}, "
                     f"mAP50-95 {map_test.get('metrics/mAP50-95(B)', float('nan')):.3f}, "
                     f"fracture AP50 {frac.get('ap50', float('nan')):.3f}")
    return "\n".join(lines) + "\n"


def plot_roc(exams: pd.DataFrame, report: dict, path: Path) -> None:
    fpr, tpr, _ = roc_curve(exams["y"], exams["score"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="0.15", lw=1.5, label=f"AUC {report['auc']['value']:.3f}")
    for name, op in report["operating_points"].items():
        label = f"threshold at val specificity {name.removeprefix('spec_')}"
        ax.plot(1 - op["specificity"]["value"], op["sensitivity"]["value"], "o", ms=5, label=label)
    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_froc(curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(curve["fp_per_image"], curve["sensitivity"], color="0.15", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 8)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Lesion sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--specificities", type=float, nargs="+", default=[0.80, 0.90, 0.95])
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    info = json.loads((a.data / "info.json").read_text())
    fid = info["fracture_class_id"]
    images = read_images(a.data / "images.csv")
    boxes = pd.read_csv(a.data / "boxes.csv", dtype={"filestem": str})

    tables = {}
    for split in ("val", "test"):
        preds = pd.read_csv(a.pred_dir / f"predictions_{split}.csv", dtype={"filestem": str})
        img = image_table(images[images["split"] == split], boxes, preds, fid)
        tables[split] = (img, exam_table(img), preds)

    val_exams = tables["val"][1]
    ops = {}
    for sp in a.specificities:
        t = threshold_at_specificity(val_exams.loc[val_exams["y"] == 0, "score"], sp)
        m = binary_metrics(val_exams["y"], val_exams["score"], t)
        ops[f"spec_{sp:.2f}"] = {
            "threshold": t,
            "target_specificity": sp,
            "val_sensitivity": m["sensitivity"],
            "val_specificity": m["specificity"],
        }
    (a.out / "operating_points.json").write_text(json.dumps(ops, indent=2))

    test_img, test_exams, test_preds = tables["test"]
    report = {"exam_level": {}, "operating_points_from_val": ops}
    for name, select in SUBGROUPS.items():
        sub = select(test_exams)
        if sub["y"].nunique() == 2:
            report["exam_level"][name] = exam_report(sub, ops, a.n_boot, a.seed)

    y_img = (test_img["n_fractures"] > 0).to_numpy()
    s_img = test_img["score"].to_numpy()
    ci = bootstrap_ci(test_img["patient_id"].to_numpy(), lambda idx: {"auc": auc(y_img[idx], s_img[idx])},
                      n_boot=a.n_boot, seed=a.seed)
    report["image_level"] = {"n_images": int(len(test_img)), "n_positive": int(y_img.sum()), "auc": ci["auc"]}

    neg = test_exams.loc[test_exams["y"] == 0, "score"]
    report["test_curve_sensitivity_at_specificity"] = {
        f"{sp:.2f}": binary_metrics(test_exams["y"], test_exams["score"],
                                    threshold_at_specificity(neg, sp))["sensitivity"]
        for sp in a.specificities
    }

    lesions, fp_scores = match_lesions(test_img["filestem"], boxes, test_preds, fid, a.iou)
    curve = froc(lesions["hit_score"].to_numpy(), fp_scores, len(test_img))
    report["lesion_level"] = {
        "iou": a.iou,
        "n_lesions": int(len(lesions)),
        "n_images": int(len(test_img)),
        "sensitivity_at_fppi": {str(x): sensitivity_at_fppi(curve, x) for x in FPPI_POINTS},
        "at_operating_points": {
            name: {
                "sensitivity": float((lesions["hit_score"] > op["threshold"]).mean()),
                "fp_per_image": float((fp_scores > op["threshold"]).sum() / len(test_img)),
            }
            for name, op in ops.items()
        },
    }

    map_path = a.pred_dir / "map_test.json"
    map_test = json.loads(map_path.read_text()) if map_path.exists() else None
    if map_test:
        report["map_test"] = map_test

    meta_cols = ["filestem", "exam_id", "patient_id", "age", "sex", "projection",
                 "initial_exam", "cast", "osteopenia", "ao"]
    lesions = lesions.merge(test_img[meta_cols], on="filestem", how="left")
    lesions.to_csv(a.out / "lesions_test.csv", index=False)
    test_exams.to_csv(a.out / "exam_scores_test.csv", index=False)
    val_exams.to_csv(a.out / "exam_scores_val.csv", index=False)
    curve.to_csv(a.out / "froc_test.csv", index=False)
    (a.out / "metrics_test.json").write_text(json.dumps(report, indent=2))
    (a.out / "results.md").write_text(markdown(report, map_test))
    plot_roc(test_exams, report["exam_level"]["all"], a.out / "roc_test.png")
    plot_froc(curve, a.out / "froc_test.png")
    print((a.out / "results.md").read_text())


if __name__ == "__main__":
    main()
