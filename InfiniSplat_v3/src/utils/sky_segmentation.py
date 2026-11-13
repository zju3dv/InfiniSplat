# Modified for SplatAnything: inference-only release.
"""ONNX sky segmentation (U^2-Net `skyseg.onnx`) enabled by --sky_seg.

The inference entry point uses --sky_seg to enable both segmentation and cloud filtering.
Per-image masks are marked with -1 in the input depth-mask tensor so the decoder can
exclude sky pixels from scale estimation and exported Gaussians.

Model contract (verified empirically against reference masks, IoU ~0.99):
  - input  : NCHW float [1, 3, 320, 320]
  - output : the fused side map [1, 1, 320, 320] in [0, 1]; HIGH value (>0.5) = SKY.
  - preprocessing: RGB, resize to 320x320, /255, then ImageNet mean/std normalization.

This is a per-image (perspective) segmenter. It intentionally does NOT implement the ERP-only
GroundingDINO+ZIM pipeline (that one must see the full panorama); it is the pragmatic backend for
ordinary perspective views where per-image inference is the only option.
"""

from pathlib import Path

import cv2
import numpy as np

_INPUT_SIZE = 320
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SkySegmenter:
    """Lazy ONNX sky segmenter. Call ``sky_mask(rgb)`` -> bool [H, W] (True = sky)."""

    def __init__(self, onnx_path: str, device: str = "cuda", prob_thresh: float = 0.5):
        self.onnx_path = str(onnx_path)
        self.prob_thresh = prob_thresh
        self.device = device
        self._sess = None
        self._iname = None
        self._oname = None
        if not Path(self.onnx_path).is_file():
            raise FileNotFoundError(f"skyseg ONNX model not found: {self.onnx_path}")

    def _ensure_session(self):
        if self._sess is not None:
            return
        import onnxruntime as ort  # local import: optional dependency

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self.device.startswith("cuda") else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(self.onnx_path, providers=providers)
        self._iname = self._sess.get_inputs()[0].name
        self._oname = self._sess.get_outputs()[0].name  # fused map (first output)
        print(f"[SkySeg] loaded {self.onnx_path} on {self._sess.get_providers()[0]}")

    def sky_mask(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: [H, W, 3] uint8 or float in [0, 1]/[0, 255]. Returns bool [H, W], True = sky.

        The 320x320 probability map is resized back to (H, W) with NEAREST so the binary mask
        stays crisp and aligns 1:1 with the per-pixel Gaussian grid.
        """
        self._ensure_session()
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            # accept [0,1] floats; scale to [0,255] for the /255 preprocessing below
            rgb = (rgb * 255.0).clip(0, 255) if rgb.max() <= 1.0 + 1e-6 else rgb
        H, W = rgb.shape[:2]

        im = cv2.resize(rgb.astype(np.float32), (_INPUT_SIZE, _INPUT_SIZE),
                        interpolation=cv2.INTER_AREA)
        im = (im / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = im.transpose(2, 0, 1)[None].astype(np.float32)              # [1, 3, 320, 320]
        prob = self._sess.run([self._oname], {self._iname: x})[0][0, 0]  # [320, 320], high = sky

        sky_small = (prob > self.prob_thresh).astype(np.uint8)
        sky = cv2.resize(sky_small, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return sky
