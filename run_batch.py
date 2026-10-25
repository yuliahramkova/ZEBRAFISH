#!/usr/bin/env python3
"""Run disjoint NAOMi seed ranges on different computers.

Python standard library only. Requires Octave and locally compiled MEX files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "profile.json"
CODE = ROOT / "simulator" / "code"
DRIVER = CODE / "run_naomi_density_image.m"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((*CODE.rglob("*.m"), *CODE.rglob("*.cpp"))):
        digest.update(str(path.relative_to(CODE)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, content: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def png_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])  # width, height


def setup_profile() -> tuple[dict, str, str]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    prior = ROOT / profile["density_prior_file"]
    if sha256(prior) != profile["density_prior_sha256"]:
        raise RuntimeError("Density prior does not match profile.json")
    return profile, sha256(PROFILE_PATH), source_hash()


def run_folder(profile: dict, seed: int, runs: Path) -> Path:
    radius = f"{float(profile['nucleus_radius_um']):.2f}".replace(".", "p")
    depth = int(profile["volume_um_yxz"][2])
    name = (
        f"naomi_density_image_{int(profile['cells'])}_nuc{radius}_seed{seed}"
        f"_z{depth}_exc{float(profile['excitation_na']):.2f}"
        f"_obj{float(profile['objective_na']):.2f}"
        f"_h{float(profile['halo_scale']):.2f}"
        f"_localbg{float(profile['local_background_level']):.2f}"
    )
    return runs / name


def image_filename(index: int, seed: int) -> str:
    return f"naomi_{index:04d}_seed{seed}.png"


def valid_state(index: int, seed: int, output: Path, profile: dict, profile_hash: str, code_hash: str) -> dict | None:
    state_file = output / "state" / f"{index:04d}.json"
    if not state_file.is_file():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        image = output / state["image"]
        expected_shape = (int(profile["image_shape_yx"][1]), int(profile["image_shape_yx"][0]))
        if (
            state["status"] == "complete"
            and state["index"] == index
            and state["seed"] == seed
            and state["profile_sha256"] == profile_hash
            and state["code_sha256"] == code_hash
            and png_shape(image) == expected_shape
            and sha256(image) == state["image_sha256"]
        ):
            return state
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return None


def write_manifest(output: Path, complete: list[dict]) -> None:
    columns = [
        "index", "seed", "cells_placed", "image", "image_sha256",
        "run_directory", "profile_sha256", "code_sha256", "finished_utc", "elapsed_seconds",
    ]
    temporary = output / "manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for state in sorted(complete, key=lambda row: row["index"]):
            writer.writerow({column: state[column] for column in columns})
    os.replace(temporary, output / "manifest.csv")


def check_setup() -> None:
    profile, profile_hash, code_hash = setup_profile()
    if shutil.which("octave") is None or shutil.which("mkoctfile") is None:
        raise RuntimeError("octave or mkoctfile is missing")
    for name in ("sample_dense_locations_exact.mex", "array_SubSubTest.mex"):
        if not (CODE / "MEX" / name).is_file():
            raise RuntimeError(f"Missing {name}; run python3 scripts/build_mex.py")
    script = (
        f"addpath(genpath('{str(CODE).replace(chr(39), chr(39)*2)}')); "
        "pkg load image; pkg load statistics; pkg load signal; "
        "assert(exist('sample_dense_locations_exact', 'file') == 3); "
        "assert(exist('array_SubSubTest', 'file') == 3); "
        "disp('Octave dependencies OK');"
    )
    result = subprocess.run(["octave", "--quiet", "--eval", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    print(f"Setup OK. Profile {profile['name']}; profile SHA256 {profile_hash[:12]}; source SHA256 {code_hash[:12]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "plan", "check"))
    parser.add_argument("--start", type=int, default=0, help="First global image index, inclusive")
    parser.add_argument("--stop", type=int, default=1000, help="Last global image index, exclusive")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "check":
        check_setup()
        return

    profile, profile_hash, code_hash = setup_profile()
    total = int(profile["total_images"])
    if not (0 <= args.start < args.stop <= total):
        parser.error(f"Require 0 <= --start < --stop <= {total}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output or ROOT / "outputs" / f"shard_{args.start:04d}_{args.stop:04d}").resolve()
    seed_base = int(profile["seed_base"])
    print(f"Global indices [{args.start}, {args.stop}); seeds [{seed_base + args.start}, {seed_base + args.stop - 1}]")
    print(f"Output: {output}")
    if args.action == "plan":
        return

    for folder in (output, output / "images", output / "runs", output / "logs", output / "state"):
        folder.mkdir(parents=True, exist_ok=True)
    assignment_file = output / "assignment.json"
    assignment = {
        "start": args.start,
        "stop": args.stop,
        "seed_base": seed_base,
        "profile_sha256": profile_hash,
        "code_sha256": code_hash,
    }
    if assignment_file.is_file():
        if json.loads(assignment_file.read_text(encoding="utf-8")) != assignment:
            raise RuntimeError("Output directory belongs to another index range or code/profile version")
    else:
        atomic_json(assignment_file, assignment)

    completed = []
    for index in range(args.start, args.stop):
        state = valid_state(index, seed_base + index, output, profile, profile_hash, code_hash)
        if state is not None:
            completed.append(state)
    if args.action == "status":
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        print(f"Complete: {len(completed)}/{args.stop - args.start}; reported status: {progress.get('status', 'not started')}")
        return

    octave = shutil.which("octave")
    if octave is None:
        raise RuntimeError("GNU Octave is missing")
    pending = [i for i in range(args.start, args.stop) if valid_state(i, seed_base + i, output, profile, profile_hash, code_hash) is None]
    print(f"Running {len(pending)} remaining images with {args.workers} processes", flush=True)

    def one(index: int) -> dict:
        seed = seed_base + index
        folder = run_folder(profile, seed, output / "runs")
        env = os.environ.copy()
        env.update({
            "NAOMI_CELL_COUNT": str(int(profile["cells"])),
            "NAOMI_NUC_RAD": str(profile["nucleus_radius_um"]),
            "NAOMI_SEED": str(seed),
            "NAOMI_DEPTH_UM": str(int(profile["volume_um_yxz"][2])),
            "NAOMI_EXC_NA": str(profile["excitation_na"]),
            "NAOMI_OBJ_NA": str(profile["objective_na"]),
            "NAOMI_HALO_SCALE": str(profile["halo_scale"]),
            "NAOMI_PRIOR_PNG": str(ROOT / profile["density_prior_file"]),
            "NAOMI_OUT_ROOT": str(output / "runs"),
            "NAOMI_EXPORT_DATASET": "1",
            "NAOMI_SKIP_PICTURES": "1",
            "NAOMI_LOCAL_BG_LEVEL": str(profile["local_background_level"]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        started = time.monotonic()
        log = output / "logs" / f"{index:04d}_seed{seed}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [octave, "--quiet", "--eval", f"run('{str(DRIVER).replace(chr(39), chr(39)*2)}')"],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=7200, check=False,
            )
        source = folder / "naomi_density_noisy_400x485.png"
        centers = folder / "naomi_centers_yx_px.csv"
        if result.returncode != 0 or not source.is_file() or not centers.is_file():
            raise RuntimeError(f"Octave failed for index {index}, seed {seed}; inspect {log}")
        expected_shape = (int(profile["image_shape_yx"][1]), int(profile["image_shape_yx"][0]))
        if png_shape(source) != expected_shape:
            raise RuntimeError(f"Wrong PNG dimensions for seed {seed}")
        with centers.open("rb") as stream:
            placed = sum(1 for _ in stream)
        if placed != int(profile["cells"]):
            raise RuntimeError(f"Only {placed}/{profile['cells']} neurons placed for seed {seed}")
        relative_image = f"images/{image_filename(index, seed)}"
        destination = output / relative_image
        temporary = destination.with_suffix(".tmp.png")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        state = {
            "status": "complete", "index": index, "seed": seed,
            "cells_placed": placed, "image": relative_image,
            "image_sha256": sha256(destination),
            "run_directory": str(folder.relative_to(output)),
            "profile_sha256": profile_hash, "code_sha256": code_hash,
            "finished_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        atomic_json(output / "state" / f"{index:04d}.json", state)
        return state

    def progress(status: str, failures: int) -> None:
        atomic_json(output / "progress.json", {
            "status": status, "assigned": args.stop - args.start,
            "complete": len(completed), "failed": failures,
            "workers": args.workers, "updated_utc": utc_now(),
        })

    progress("running", 0)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                state = future.result()
                completed.append(state)
                (output / "state" / f"{index:04d}.error.json").unlink(missing_ok=True)
                print(f"[{len(completed)}/{args.stop - args.start}] index {index}, seed {state['seed']} complete", flush=True)
            except Exception as exc:
                failed += 1
                atomic_json(output / "state" / f"{index:04d}.error.json", {
                    "index": index, "seed": seed_base + index,
                    "error": str(exc), "updated_utc": utc_now(),
                })
                print(f"Index {index} ERROR: {exc}", file=sys.stderr, flush=True)
            write_manifest(output, completed)
            progress("running", failed)
    progress("complete" if len(completed) == args.stop - args.start else "needs_retry", failed)
    print(f"Finished: {len(completed)}/{args.stop - args.start} complete, {failed} errors")


if __name__ == "__main__":
    main()
