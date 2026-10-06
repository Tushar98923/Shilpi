"""Vendor the shared modules into the add-on and zip it for Blender's "Install from Disk".

The add-on must use byte-identical copies of the scene serializer, output
formats and op executor that built the training data, so it copies them
rather than keeping its own versions.

Usage:
    python scripts/build_addon.py        # -> addon/shilpi.zip
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "addon" / "shilpi"
VENDORED = ["scene_serializer.py", "output_formats.py", "op_executor.py"]
VENDORED_DATA = ["op_schema.json", "selection_free_ops.json"]
VENDORED_SCRIPTS = ["asr_server.py"]  # the speech server the add-on starts with Blender's Python


def main():
    vendor = ADDON / "vendor"
    vendor.mkdir(exist_ok=True)
    (vendor / "__init__.py").write_text('"""Copied from shared/ by scripts/build_addon.py -- do not edit here."""\n', encoding="utf-8")
    for name in VENDORED:
        shutil.copyfile(ROOT / "shared" / name, vendor / name)
    for name in VENDORED_DATA:  # built by the pipeline's build_op_schema.py
        shutil.copyfile(ROOT / "shared" / "data" / name, vendor / name)
    for name in VENDORED_SCRIPTS:
        shutil.copyfile(ROOT / "scripts" / name, vendor / name)

    zip_path = ROOT / "addon" / "shilpi.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ADDON.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(ADDON))
    print(f"Built {zip_path}")


if __name__ == "__main__":
    main()
