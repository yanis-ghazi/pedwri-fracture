import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PRED_COLUMNS = ["filestem", "cls", "conf", "x1", "y1", "x2", "y2"]


def _codes_union(values: pd.Series) -> str:
    codes = {c.strip() for v in values for c in str(v).split(";") if c.strip()}
    return ";".join(sorted(codes))


def image_table(images: pd.DataFrame, boxes: pd.DataFrame, preds: pd.DataFrame,
                fracture_id: int) -> pd.DataFrame:
    n_gt = boxes[boxes["cls"] == fracture_id].groupby("filestem").size()
    score = preds[preds["cls"] == fracture_id].groupby("filestem")["conf"].max()
    img = images.copy()
    img["n_fractures"] = img["filestem"].map(n_gt).fillna(0).astype(int)
    img["score"] = img["filestem"].map(score).fillna(0.0)
    return img


def exam_table(img: pd.DataFrame) -> pd.DataFrame:
    exams = img.groupby("exam_id").agg(
        patient_id=("patient_id", "first"),
        y=("n_fractures", lambda s: int((s > 0).any())),
        score=("score", "max"),
        age=("age", "min"),
        sex=("sex", "first"),
        initial_exam=("initial_exam", "any"),
        cast=("cast", "any"),
        diagnosis_uncertain=("diagnosis_uncertain", "any"),
        osteopenia=("osteopenia", "any"),
        metal=("metal", "any"),
        ao=("ao", _codes_union),
        n_images=("filestem", "size"),
    )
    return exams.reset_index()


def threshold_at_specificity(neg_scores, target: float) -> float:
    s = np.sort(np.asarray(neg_scores, dtype=float))
    k = min(max(int(np.ceil(target * len(s) - 1e-9)), 1), len(s))
    return float(s[k - 1])


def binary_metrics(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(score) > threshold
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    div = lambda a, b: a / b if b else float("nan")
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def auc(y, score) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score))


def bootstrap_ci(groups, stat_fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    codes, _ = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    members = [order[s:s + c] for s, c in zip(starts, counts)]

    point = stat_fn(np.arange(len(codes)))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        samples.append(stat_fn(np.concatenate([members[i] for i in pick])))

    out = {}
    for key, value in point.items():
        vals = np.array([s[key] for s in samples], dtype=float)
        vals = vals[np.isfinite(vals)]
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan)
        out[key] = {"value": float(value), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def gt_xyxy(boxes: pd.DataFrame) -> np.ndarray:
    cx, cy, w, h = (boxes[c].to_numpy(float) for c in ("cx", "cy", "w", "h"))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def match_lesions(stems, gt: pd.DataFrame, preds: pd.DataFrame, fracture_id: int,
                  iou_thr: float = 0.3) -> tuple[pd.DataFrame, np.ndarray]:
    gt = gt[gt["cls"] == fracture_id]
    preds = preds[preds["cls"] == fracture_id]
    gt_by = dict(tuple(gt.groupby("filestem")))
    pr_by = dict(tuple(preds.groupby("filestem")))

    lesions, fp_scores = [], []
    for stem in stems:
        g = gt_by.get(stem)
        p = pr_by.get(stem)
        g_xy = gt_xyxy(g) if g is not None else np.zeros((0, 4))
        p_xy = p[["x1", "y1", "x2", "y2"]].to_numpy(float) if p is not None else np.zeros((0, 4))
        p_sc = p["conf"].to_numpy(float) if p is not None else np.zeros(0)

        hit = np.full(len(g_xy), -1.0)
        used = np.zeros(len(g_xy), dtype=bool)
        ious = iou_matrix(p_xy, g_xy) if len(g_xy) and len(p_xy) else None
        for i in np.argsort(-p_sc):
            if ious is not None:
                cand = np.where(used, -1.0, ious[i])
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    used[j] = True
                    hit[j] = p_sc[i]
                    continue
            fp_scores.append(p_sc[i])

        for j, box in enumerate(g_xy):
            w, h = box[2] - box[0], box[3] - box[1]
            lesions.append((stem, j, *box, w, h, hit[j]))

    cols = ["filestem", "lesion", "x1", "y1", "x2", "y2", "w", "h", "hit_score"]
    return pd.DataFrame(lesions, columns=cols), np.asarray(fp_scores, dtype=float)


def froc(hit_scores: np.ndarray, fp_scores: np.ndarray, n_images: int) -> pd.DataFrame:
    hit_scores = np.asarray(hit_scores, dtype=float)
    thresholds = np.unique(np.r_[hit_scores[hit_scores >= 0], fp_scores])[::-1]
    hs, fs = np.sort(hit_scores), np.sort(fp_scores)
    tp = len(hs) - np.searchsorted(hs, thresholds, side="left")
    fp = len(fs) - np.searchsorted(fs, thresholds, side="left")
    return pd.DataFrame({
        "threshold": thresholds,
        "sensitivity": tp / max(len(hs), 1),
        "fp_per_image": fp / max(n_images, 1),
    })


def sensitivity_at_fppi(curve: pd.DataFrame, fppi: float) -> float:
    ok = curve[curve["fp_per_image"] <= fppi]
    return float(ok["sensitivity"].max()) if len(ok) else 0.0
