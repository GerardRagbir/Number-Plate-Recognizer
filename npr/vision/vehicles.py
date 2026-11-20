"""Vehicle detection + instance masks: a YOLO-seg Core ML model driven through Apple Vision.

Vision hands the CVPixelBuffer to Core ML directly (scaling on the GPU, inference on the Neural
Engine / Metal), so the only CPU work is decoding the raw head output below.
"""
from __future__ import annotations

from pathlib import Path

import CoreML
import cv2
import numpy as np
import Vision
from Foundation import NSURL

from ..tracking.tracker import Detection

_DTYPES = {
    CoreML.MLMultiArrayDataTypeFloat32: np.float32,
    CoreML.MLMultiArrayDataTypeFloat16: np.float16,
    CoreML.MLMultiArrayDataTypeDouble: np.float64,
    CoreML.MLMultiArrayDataTypeInt32: np.int32,
}


def multiarray_to_numpy(ma) -> np.ndarray:
    shape = tuple(int(x) for x in ma.shape())
    dtype = np.dtype(_DTYPES[ma.dataType()])
    strides = tuple(int(s) * dtype.itemsize for s in ma.strides())
    n = int(np.prod(shape))
    span = sum((d - 1) * s for d, s in zip(shape, strides)) + dtype.itemsize if n else 0
    buf = ma.dataPointer().as_buffer(span)
    arr = np.lib.stride_tricks.as_strided(np.frombuffer(buf, dtype=dtype, count=span // dtype.itemsize),
                                          shape=shape, strides=strides)
    return np.array(arr, dtype=np.float32)  # own the memory before the MLMultiArray goes away


def load_coreml(path: Path, units: int = CoreML.MLComputeUnitsAll):
    """Load (compiling and caching if needed) an .mlpackage / .mlmodelc."""
    path = Path(path)
    compiled = path if path.suffix == ".mlmodelc" else path.with_suffix(".mlmodelc")
    if not compiled.exists() or compiled.stat().st_mtime < path.stat().st_mtime:
        tmp, err = CoreML.MLModel.compileModelAtURL_error_(NSURL.fileURLWithPath_(str(path)), None)
        if tmp is None:
            raise RuntimeError(f"Core ML compile failed: {err}")
        import shutil

        shutil.rmtree(compiled, ignore_errors=True)
        shutil.move(str(tmp.path()), str(compiled))
    cfg = CoreML.MLModelConfiguration.alloc().init()
    cfg.setComputeUnits_(units)
    model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
        NSURL.fileURLWithPath_(str(compiled)), cfg, None)
    if model is None:
        raise RuntimeError(f"Core ML load failed: {err}")
    return model


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        iou = w * h / (area[i] + area[rest] - w * h + 1e-9)
        order = rest[iou <= thresh]
    return np.array(keep, dtype=int)


class VehicleSegmenter:
    def __init__(self, model_path: Path, classes: dict[int, str], conf: float = 0.15, iou: float = 0.55):
        self.classes, self.conf, self.iou = classes, conf, iou
        self._cls_ids = np.array(sorted(classes))
        self.mlmodel = load_coreml(model_path)
        self.vnmodel, err = Vision.VNCoreMLModel.modelForMLModel_error_(self.mlmodel, None)
        if self.vnmodel is None:
            raise RuntimeError(f"VNCoreMLModel: {err}")
        desc = self.mlmodel.modelDescription()
        inp = next(iter(desc.inputDescriptionsByName().values()))
        c = inp.imageConstraint()
        self.size = int(c.pixelsWide()) if c is not None else 640

    def request(self):
        """A VNCoreMLRequest is not thread-safe: each pipeline keeps its own."""
        req = Vision.VNCoreMLRequest.alloc().initWithModel_(self.vnmodel)
        req.setImageCropAndScaleOption_(Vision.VNImageCropAndScaleOptionScaleFit)
        return req

    def detect(self, req, handler, width: int, height: int) -> list[Detection]:
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"vehicle model: {err}")
        outs = [multiarray_to_numpy(o.featureValue().multiArrayValue()) for o in req.results() or []]
        proto = next((o for o in outs if o.ndim == 4), None)
        pred = next((o for o in outs if o.ndim == 3), None)
        if pred is None:
            return []
        pred = pred[0]
        nm = proto.shape[1] if proto is not None else 0
        if pred.shape[-1] == 6 + nm and pred.shape[0] != 6 + nm:  # end-to-end head: x1 y1 x2 y2 conf cls [coef]
            boxes, scores, cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)
            coef = pred[:, 6:]
            m = (scores >= self.conf) & np.isin(cls, self._cls_ids)
            boxes, scores, cls, coef = boxes[m], scores[m], cls[m], coef[m]
        else:  # classic head: (4 + classes + coef, anchors)
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T
            nc = pred.shape[1] - 4 - nm
            sub = pred[:, 4 + self._cls_ids[self._cls_ids < nc]]
            scores = sub.max(1)
            m = scores >= self.conf
            pred, sub, scores = pred[m], sub[m], scores[m]
            cls = self._cls_ids[sub.argmax(1)] if len(sub) else np.zeros(0, int)
            cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            coef = pred[:, 4 + nc:]
        if not len(boxes):
            return []
        keep = _nms(boxes, scores, self.iou)  # class-agnostic: a pickup is not a car *and* a truck
        boxes, scores, cls, coef = boxes[keep], scores[keep], cls[keep], coef[keep]

        # undo Vision's centred "scale to fit" letterbox
        S = float(self.size)
        scale = S / max(width, height)
        pad_x, pad_y = (S - width * scale) / 2, (S - height * scale) / 2
        dets: list[Detection] = []
        for i in range(len(boxes)):
            b = boxes[i]
            src = np.array([(b[0] - pad_x) / scale, (b[1] - pad_y) / scale,
                            (b[2] - pad_x) / scale, (b[3] - pad_y) / scale])
            src[[0, 2]] = src[[0, 2]].clip(0, width)
            src[[1, 3]] = src[[1, 3]].clip(0, height)
            if src[2] - src[0] < 4 or src[3] - src[1] < 4:
                continue
            poly = mask = None
            if proto is not None and nm:
                mask, poly = self._mask(coef[i], proto[0], b, S, scale, pad_x, pad_y)
            dets.append(Detection(src, float(scores[i]), int(cls[i]), polygon=poly, mask=mask))
        return dets

    @staticmethod
    def _mask(coef, proto, box, S, scale, pad_x, pad_y):
        """Instance mask for one detection → (bool mask cropped to the box, outline in source px)."""
        mh, mw = proto.shape[1:]
        r = mw / S
        x1, y1 = max(int(box[0] * r), 0), max(int(box[1] * r), 0)
        x2, y2 = min(int(np.ceil(box[2] * r)), mw), min(int(np.ceil(box[3] * r)), mh)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None, None
        logits = np.tensordot(coef, proto[:, y1:y2, x1:x2], axes=1)
        mask = logits > 0  # sigmoid(x) > 0.5
        up = cv2.resize(mask.astype(np.uint8) * 255, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((up > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, None
        c = max(contours, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            return mask, None
        c = c / 4 + (x1, y1)  # back to proto space
        c = (c / r - (pad_x, pad_y)) / scale  # → model input → source pixels
        return mask, c
