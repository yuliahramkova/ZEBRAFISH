#!/usr/bin/env python3
"""Compile the two MEX functions needed by the nucleus-only NAOMi profile."""

import shutil
import subprocess
from pathlib import Path


root = Path(__file__).resolve().parent.parent
mex_dir = root / "simulator" / "code" / "MEX"
compiler = shutil.which("mkoctfile")
if compiler is None:
    raise SystemExit("mkoctfile not found. Install GNU Octave with development tools first.")
for name in ("sample_dense_locations_exact", "array_SubSubTest"):
    print(f"Compiling {name}.cpp for this machine...", flush=True)
    subprocess.run([compiler, "--mex", "-O3", "-o", str(mex_dir / name), str(mex_dir / f"{name}.cpp")], check=True)
print("MEX build complete.")
