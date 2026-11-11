import pandas as pd
from sklearn.model_selection import train_test_split

from pedwri.metadata import age_bin


def patient_table(images: pd.DataFrame, boxes: pd.DataFrame, fracture_id: int) -> pd.DataFrame:
    fractured = set(boxes.loc[boxes["cls"] == fracture_id, "filestem"])
    df = images.assign(fracture=images["filestem"].isin(fractured))
    pt = df.groupby("patient_id").agg(
        fracture=("fracture", "any"),
        age=("age", "min"),
        n_exams=("exam_id", "nunique"),
        n_images=("filestem", "size"),
    )
    pt["stratum"] = pt["fracture"].astype(int).astype(str) + "_" + age_bin(pt["age"]).to_numpy()
    return pt


def _robust_strata(strata: pd.Series, fallback: pd.Series, min_count: int) -> pd.Series:
    counts = strata.map(strata.value_counts())
    return strata.where(counts >= min_count, fallback.astype(str))


def split_patients(pt: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = 42) -> pd.Series:
    strata = _robust_strata(pt["stratum"], pt["fracture"].astype(int), min_count=8)
    train_ids, rest_ids = train_test_split(
        pt.index, test_size=val_frac + test_frac, stratify=strata, random_state=seed
    )
    rest_strata = _robust_strata(strata.loc[rest_ids], pt.loc[rest_ids, "fracture"].astype(int), 4)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=test_frac / (val_frac + test_frac), stratify=rest_strata, random_state=seed
    )
    split = pd.Series("train", index=pt.index, name="split")
    split.loc[val_ids] = "val"
    split.loc[test_ids] = "test"
    return split


def split_images(images: pd.DataFrame, boxes: pd.DataFrame, fracture_id: int, val_frac: float = 0.15,
                 test_frac: float = 0.15, seed: int = 42) -> pd.Series:
    fractured = images["filestem"].isin(set(boxes.loc[boxes["cls"] == fracture_id, "filestem"])).astype(int)
    train_idx, rest_idx = train_test_split(
        images.index, test_size=val_frac + test_frac, stratify=fractured, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_frac / (val_frac + test_frac), stratify=fractured.loc[rest_idx],
        random_state=seed,
    )
    split = pd.Series("train", index=images.index, name="split")
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split


def split_exams(images: pd.DataFrame, boxes: pd.DataFrame, fracture_id: int, val_frac: float = 0.15,
                test_frac: float = 0.15, seed: int = 42) -> pd.Series:
    fractured = images["filestem"].isin(set(boxes.loc[boxes["cls"] == fracture_id, "filestem"]))
    exams = fractured.groupby(images["exam_id"]).any().astype(int)
    train_ids, rest_ids = train_test_split(
        exams.index, test_size=val_frac + test_frac, stratify=exams, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=test_frac / (val_frac + test_frac), stratify=exams.loc[rest_ids], random_state=seed
    )
    split = pd.Series("train", index=exams.index)
    split.loc[val_ids] = "val"
    split.loc[test_ids] = "test"
    return images["exam_id"].map(split).rename("split")


def leaked_patients(images: pd.DataFrame) -> int:
    return int((images.groupby("patient_id")["split"].nunique() > 1).sum())


def assert_no_leak(images: pd.DataFrame) -> None:
    n = leaked_patients(images)
    if n:
        raise AssertionError(f"{n} patients appear in more than one split")
