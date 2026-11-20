"""``python -m npr serve`` runs the dashboard; ``python -m npr setup`` fetches and builds the models."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .config import settings


def setup(_args) -> int:
    settings.ensure()
    dest = settings.models_dir / f"{settings.vehicle_model}.mlpackage"
    if dest.exists():
        print(f"✓ vehicle model already built: {dest}")
    else:
        uv = shutil.which("uv")
        if uv is None:
            print("`uv` is needed to build the vehicle model: https://docs.astral.sh/uv/", file=sys.stderr)
            return 1
        script = Path(__file__).parent / "tools" / "export_vehicle_model.py"
        print(f"Converting {settings.vehicle_model} to Core ML (one-off; downloads PyTorch into a temporary environment)…")
        r = subprocess.run([uv, "run", "--no-project", str(script), settings.vehicle_model, str(settings.models_dir)])
        if r.returncode or not dest.exists():
            print("vehicle model conversion failed", file=sys.stderr)
            return 1
        print(f"✓ {dest}")
    print("Fetching the plate detector…")
    from .vision.plates import PlateDetector

    PlateDetector(settings.plate_model, settings.plate_conf)
    print("✓ plate detector ready\n\nRun the dashboard with:  uv run python -m npr serve")
    return 0


def serve(args) -> int:
    import uvicorn

    from .server.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", ws_max_size=1 << 26)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="npr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8314)
    s.set_defaults(fn=serve)
    sub.add_parser("setup", help="download / convert models").set_defaults(fn=setup)
    args = ap.parse_args()
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
