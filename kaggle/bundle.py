"""Zip the code + mini data needed by kaggle/encode.py into kaggle/bundle.zip."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("src/blended_rag", "data/mini", "kaggle/encode.py")

with zipfile.ZipFile(ROOT / "kaggle" / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        path = ROOT / item
        for f in [path] if path.is_file() else sorted(path.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(ROOT))
print(f"wrote {ROOT / 'kaggle' / 'bundle.zip'}")
