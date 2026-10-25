#!/usr/bin/env python3
"""Generate one tiny 20-cell frame to test the full Octave pipeline."""

import os
import shutil
import subprocess
from pathlib import Path


root = Path(__file__).resolve().parent.parent
driver = root / "simulator" / "code" / "run_naomi_density_image.m"
octave = shutil.which("octave")
if octave is None:
    raise SystemExit("GNU Octave not found")
env = os.environ.copy()
env.update({
    "NAOMI_CELL_COUNT": "20",
    "NAOMI_NUC_RAD": "1.6",
    "NAOMI_SEED": "20262001",
    "NAOMI_DEPTH_UM": "10",
    "NAOMI_EXC_NA": "0.6",
    "NAOMI_OBJ_NA": "0.8",
    "NAOMI_HALO_SCALE": "0.75",
    "NAOMI_PRIOR_PNG": str(root / "density_prior_seed20262001.png"),
    "NAOMI_OUT_ROOT": str(root / "outputs" / "smoke"),
    "NAOMI_EXPORT_DATASET": "1",
    "NAOMI_SKIP_PICTURES": "1",
    "NAOMI_LOCAL_BG_LEVEL": "10",
})
subprocess.run([octave, "--quiet", "--eval", f"run('{str(driver).replace(chr(39), chr(39)*2)}')"], cwd=root, env=env, check=True)
frames = list((root / "outputs" / "smoke").rglob("naomi_density_noisy_400x485.png"))
if not frames:
    raise SystemExit("No PNG produced by the smoke test")
print(f"Smoke test OK: {frames[0]}")
