import ast
from pathlib import Path

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int, color: int = 114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


class Detector:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = inp.shape[2] if isinstance(inp.shape[2], int) else 1024
        self.names = self._names()

    def _names(self) -> dict[int, str]:
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return {}
        parsed = ast.literal_eval(raw)
        items = parsed.items() if isinstance(parsed, dict) else enumerate(parsed)
        return {int(k): str(v) for k, v in items}

    def __call__(self, img: np.ndarray, conf: float = 0.001, iou: float = 0.6, max_det: int = 100) -> list[dict]:
        lb, r, (left, top) = letterbox(img, self.imgsz)
        if lb.ndim == 2:
            lb = np.repeat(lb[..., None], 3, axis=2)
        x = np.ascontiguousarray(lb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0].T

        cls_scores = out[:, 4:]
        cls = cls_scores.argmax(axis=1)
        score = cls_scores[np.arange(len(cls)), cls]
        keep = score > conf
        if not keep.any():
            return []
        b, score, cls = out[keep, :4], score[keep], cls[keep]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
        idx = nms(xyxy + cls[:, None] * 7680.0, score, iou)[:max_det]
        xyxy, score, cls = xyxy[idx], score[idx], cls[idx]

        h, w = img.shape[:2]
        xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - left) / r, 0, w)
        xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - top) / r, 0, h)
        return [
            {"label": self.names.get(int(c), str(int(c))), "score": float(s), "box": [float(v) for v in bb]}
            for bb, s, c in zip(xyxy, score, cls)
        ]
