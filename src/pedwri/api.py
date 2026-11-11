import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from pedwri import __version__
from pedwri.detector import Detector
from pedwri.preprocess import decode, standardize

MAX_FILES = 8


def create_app(model_path=None, operating_points=None, operating_point=None,
               max_side=None, box_score=None) -> FastAPI:
    model_path = Path(model_path or os.getenv("PEDWRI_MODEL", "models/model.onnx"))
    ops_path = Path(operating_points or os.getenv("PEDWRI_OPERATING_POINTS", "models/operating_points.json"))
    op_name = operating_point or os.getenv("PEDWRI_OPERATING_POINT", "spec_0.90")
    max_side = int(max_side or os.getenv("PEDWRI_MAX_SIDE", 1024))
    box_score = float(box_score if box_score is not None else os.getenv("PEDWRI_BOX_SCORE", 0.25))
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ops = json.loads(ops_path.read_text())
        if op_name not in ops:
            raise RuntimeError(f"operating point {op_name} not in {ops_path}, available: {list(ops)}")
        state["op"] = {"name": op_name, **ops[op_name]}
        state["detector"] = Detector(model_path)
        yield
        state.clear()

    app = FastAPI(title="pedwri", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health():
        op = state["op"]
        return {"status": "ok", "model": model_path.name, "operating_point": op["name"],
                "threshold": op["threshold"], "version": __version__}

    @app.post("/predict")
    def predict(files: list[UploadFile] = File(...)):
        if len(files) > MAX_FILES:
            raise HTTPException(413, f"at most {MAX_FILES} images per exam")
        detector, op = state["detector"], state["op"]
        results, exam_score = [], 0.0
        for f in files:
            try:
                raw = decode(f.file.read())
            except Exception as exc:
                raise HTTPException(415, f"{f.filename}: unreadable image or DICOM") from exc
            h, w = raw.shape[:2]
            std = standardize(raw, max_side)
            scale = w / std.shape[1]
            dets = detector(std)
            score = max((d["score"] for d in dets if d["label"] == "fracture"), default=0.0)
            exam_score = max(exam_score, score)
            results.append({
                "filename": f.filename,
                "width": int(w),
                "height": int(h),
                "fracture_score": round(score, 4),
                "detections": [
                    {"label": d["label"], "score": round(d["score"], 4),
                     "box": [round(v * scale, 1) for v in d["box"]]}
                    for d in dets if d["score"] >= box_score
                ],
            })
        return {
            "exam_score": round(exam_score, 4),
            "fracture_suspected": exam_score > op["threshold"],
            "operating_point": {
                "name": op["name"],
                "threshold": op["threshold"],
                "target_specificity": op.get("target_specificity"),
            },
            "images": results,
            "version": __version__,
            "intended_use": "research prototype, not a medical device",
        }

    return app


app = create_app()
