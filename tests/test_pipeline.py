import json

import pandas as pd

from pedwri import errors, evaluate, prepare
from synthetic import fake_predictions, make_dataset


def test_prepare_evaluate_errors(tmp_path):
    raw = make_dataset(tmp_path / "raw")
    out = tmp_path / "prepared"
    prepare.main(["--root", str(raw), "--out", str(out), "--max-side", "128", "--workers", "1"])

    images = pd.read_csv(out / "images.csv", dtype={"patient_id": str})
    assert images.groupby("patient_id")["split"].nunique().max() == 1
    assert set(images["split"]) == {"train", "val", "test"}
    assert images[["width", "height"]].max().max() <= 128
    info = json.loads((out / "info.json").read_text())
    assert info["fracture_class_id"] == 3
    for split in ("train", "val", "test"):
        n_img = len(list((out / "images" / split).glob("*.png")))
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        assert n_img == n_lbl == info["counts"][split]["images"]

    preds = tmp_path / "preds"
    preds.mkdir()
    for split in ("val", "test"):
        fake_predictions(out, split, seed=1).to_csv(preds / f"predictions_{split}.csv", index=False)
    res = tmp_path / "results"
    evaluate.main(["--data", str(out), "--pred-dir", str(preds), "--out", str(res), "--n-boot", "200"])

    ops = json.loads((res / "operating_points.json").read_text())
    for op in ops.values():
        assert op["val_specificity"] >= op["target_specificity"]
    report = json.loads((res / "metrics_test.json").read_text())
    assert 0.5 < report["exam_level"]["all"]["auc"]["value"] <= 1.0
    assert report["image_level"]["n_images"] == info["counts"]["test"]["images"]
    for f in ("results.md", "roc_test.png", "froc_test.png", "lesions_test.csv", "exam_scores_test.csv"):
        assert (res / f).exists()

    errors.main(["--results", str(res), "--images", str(out / "images" / "test"), "--min-n", "3"])
    assert (res / "errors" / "summary.md").exists()
    assert (res / "errors" / "missed_exams.png").exists()
    assert (res / "errors" / "missed_exams_by_fracture_pattern.csv").exists()


def test_image_resplit_leaks_patients(tmp_path):
    from pedwri import resplit

    raw = make_dataset(tmp_path / "raw", n_patients=80)
    out = tmp_path / "prepared"
    prepare.main(["--root", str(raw), "--out", str(out), "--max-side", "64", "--workers", "1"])
    leaky = tmp_path / "leaky"
    resplit.main(["--data", str(out), "--out", str(leaky), "--unit", "image"])

    images = pd.read_csv(leaky / "images.csv", dtype={"patient_id": str})
    assert (images.groupby("patient_id")["split"].nunique() > 1).sum() > 0
    info = json.loads((leaky / "info.json").read_text())
    assert info["split_unit"] == "image"
    for split in ("train", "val", "test"):
        files = list((leaky / "images" / split).glob("*.png"))
        assert len(files) == info["counts"][split]["images"]
        assert all(f.resolve().exists() for f in files)


def test_exam_resplit_keeps_exams_whole(tmp_path):
    from pedwri import resplit

    raw = make_dataset(tmp_path / "raw", n_patients=80)
    out = tmp_path / "prepared"
    prepare.main(["--root", str(raw), "--out", str(out), "--max-side", "64", "--workers", "1"])
    by_exam = tmp_path / "by_exam"
    resplit.main(["--data", str(out), "--out", str(by_exam), "--unit", "exam"])

    images = pd.read_csv(by_exam / "images.csv", dtype={"patient_id": str, "exam_id": str})
    assert images.groupby("exam_id")["split"].nunique().max() == 1
    assert (images.groupby("patient_id")["split"].nunique() > 1).sum() > 0
    assert json.loads((by_exam / "info.json").read_text())["split_unit"] == "exam"

    from pedwri import seen

    preds = tmp_path / "preds"
    preds.mkdir()
    for split in ("val", "test"):
        fake_predictions(by_exam, split, seed=1).to_csv(preds / f"predictions_{split}.csv", index=False)
    res = tmp_path / "results"
    evaluate.main(["--data", str(by_exam), "--pred-dir", str(preds), "--out", str(res), "--n-boot", "100"])
    seen.main(["--data", str(by_exam), "--results", str(res), "--n-boot", "100"])
    report = json.loads((res / "seen_patients.json").read_text())
    groups = report["groups"]
    n_test = images.loc[images["split"] == "test", "exam_id"].nunique()
    assert groups["seen"]["n_exams"] + groups["unseen"]["n_exams"] == n_test
    assert (res / "seen_patients.md").exists()
