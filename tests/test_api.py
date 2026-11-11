import io
import json

import cv2
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from pedwri.api import create_app
from pedwri.detector import Detector
from pedwri.metadata import DEFAULT_NAMES
from pedwri.preprocess import decode

IMGSZ = 64


def dummy_model(path, detections):
    out = np.zeros((1, 4 + len(DEFAULT_NAMES), len(detections) + 1), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(detections):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + c, i] = s
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(out))],
        "dummy",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IMGSZ, IMGSZ])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    helper.set_model_props(model, {"names": str(dict(enumerate(DEFAULT_NAMES)))})
    onnx.save(model, str(path))
    return path


@pytest.fixture
def model_path(tmp_path):
    return dummy_model(tmp_path / "model.onnx", [(32, 32, 16, 8, 3, 0.9), (10, 10, 4, 4, 8, 0.95)])


def test_detector_undoes_letterbox(model_path):
    det = Detector(model_path)
    assert det.imgsz == IMGSZ and det.names[3] == "fracture"
    out = det(np.zeros((64, 128), np.uint8))
    frac = [d for d in out if d["label"] == "fracture"][0]
    assert np.allclose(frac["box"], [48, 24, 80, 40])


def dicom_bytes(arr):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_dicom_monochrome1_is_inverted():
    arr = np.tile(np.arange(0, 4000, 40, dtype=np.uint16), (10, 1))
    img = decode(dicom_bytes(arr))
    assert img.shape == arr.shape
    assert img[0, 0] > img[0, -1]


@pytest.fixture
def client(tmp_path, model_path):
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps({"spec_0.90": {"threshold": 0.5, "target_specificity": 0.9}}))
    app = create_app(model_path=model_path, operating_points=ops, max_side=128)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["threshold"] == 0.5


def test_predict_exam_with_png_and_dicom(client):
    png = cv2.imencode(".png", (np.random.default_rng(0).random((128, 256)) * 65535).astype(np.uint16))[1]
    dcm = dicom_bytes(np.random.default_rng(1).integers(0, 4000, (64, 128)))
    r = client.post("/predict", files=[
        ("files", ("ap.png", png.tobytes(), "image/png")),
        ("files", ("lat.dcm", dcm, "application/dicom")),
    ])
    assert r.status_code == 200
    body = r.json()
    assert body["exam_score"] == 0.9 and body["fracture_suspected"] is True
    ap = body["images"][0]
    assert ap["width"] == 256 and ap["height"] == 128
    frac = [d for d in ap["detections"] if d["label"] == "fracture"][0]
    assert frac["box"] == [96.0, 48.0, 160.0, 80.0]


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files=[("files", ("x.png", b"not an image", "image/png"))])
    assert r.status_code == 415
