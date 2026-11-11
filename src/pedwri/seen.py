import argparse
import json
from pathlib import Path

import pandas as pd

from pedwri.evaluate import fmt
from pedwri.metadata import read_images
from pedwri.metrics import auc, binary_metrics, bootstrap_ci


def compare(exams: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> dict:
    y = exams["y"].to_numpy()
    s = exams["score"].to_numpy()

    def stats(idx):
        m = binary_metrics(y[idx], s[idx], threshold)
        return {"auc": auc(y[idx], s[idx]), "sensitivity": m["sensitivity"], "specificity": m["specificity"]}

    ci = bootstrap_ci(exams["patient_id"].to_numpy(), stats, n_boot=n_boot, seed=seed)
    return {"n_exams": int(len(exams)), "n_positive": int(y.sum()), "n_patients": int(exams["patient_id"].nunique()),
            **ci}


def main(argv=None):
    p = argparse.ArgumentParser(description="Test exams of patients also present in train vs unseen patients")
    p.add_argument("--data", type=Path, required=True, help="prepared dataset whose split lets patients leak")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--op", default="spec_0.90")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    images = read_images(a.data / "images.csv")
    in_train = set(images.loc[images["split"] == "train", "patient_id"])
    exams = pd.read_csv(a.results / "exam_scores_test.csv", dtype={"patient_id": str, "exam_id": str},
                        keep_default_na=False)
    exams["initial_exam"] = exams["initial_exam"].astype(str).str.lower() == "true"
    exams["seen"] = exams["patient_id"].isin(in_train)
    t = json.loads((a.results / "operating_points.json").read_text())[a.op]["threshold"]

    groups = {
        "seen": exams[exams["seen"]],
        "unseen": exams[~exams["seen"]],
        "seen, initial exams": exams[exams["seen"] & exams["initial_exam"]],
        "unseen, initial exams": exams[~exams["seen"] & exams["initial_exam"]],
    }
    report = {"operating_point": a.op, "threshold": t,
              "groups": {k: compare(g, t, a.n_boot, a.seed) for k, g in groups.items() if g["y"].nunique() == 2}}
    (a.results / "seen_patients.json").write_text(json.dumps(report, indent=2))

    lines = [f"Test exams of patients with other exams in train (seen) or not (unseen), threshold {a.op} "
             f"({t:.3f}).\n",
             "| Exams | n | Positive | AUC | Sensitivity | Specificity |",
             "|---|---|---|---|---|---|"]
    for name, r in report["groups"].items():
        lines.append(f"| {name} | {r['n_exams']} | {r['n_positive']} | {fmt(r['auc'])} | "
                     f"{fmt(r['sensitivity'])} | {fmt(r['specificity'])} |")
    (a.results / "seen_patients.md").write_text("\n".join(lines) + "\n")
    print((a.results / "seen_patients.md").read_text())


if __name__ == "__main__":
    main()
