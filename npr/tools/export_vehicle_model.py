# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["ultralytics>=8.4", "coremltools>=9", "numpy<2.4"]
# ///
"""Convert an Ultralytics YOLO-seg checkpoint to a Core ML .mlpackage.

Run by ``npr setup`` through ``uv run``, in its own throwaway environment: coremltools has no
Python 3.14 build yet, and the app has no reason to carry PyTorch around at runtime.

    uv run --no-project export_vehicle_model.py yolo26s-seg /path/to/models
"""
import shutil
import sys
from pathlib import Path


def patch_coremltools() -> None:
    """coremltools calls int()/float() on 1-element arrays, which current NumPy rejects."""
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    original = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and not isinstance(x.val, dtype) and np.ndim(x.val) > 0:
            context.add(mb.const(val=dtype(np.asarray(x.val).reshape(-1)[0]), name=node.name), node.name)
            return
        original(context, node, dtype, dtype_name)

    ops._cast = _cast


def main() -> int:
    name, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.mlpackage"
    from ultralytics import YOLO

    patch_coremltools()
    exported = YOLO(str(out_dir / f"{name}.pt")).export(format="coreml", imgsz=640, nms=False, half=True)
    exported = Path(exported)
    if exported.resolve() != dest.resolve():  # the exporter normally writes next to the weights already
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(exported), str(dest))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
