"""Does more CPU per GPU let us pack more predictors onto one L4?

Runs aiddd_pipeline.py once per (model, predictors-per-GPU) setting on the same
ligands and one GPU, and prints seconds per ligand and GPU utilization for each.
Run it twice, on an 8-vCPU and a 16-vCPU GPU worker, to separate the effect of
packing from the effect of the CPU available to feed it:

  anyscale job submit -f job_pack.yaml --name pack8
  anyscale job submit -f job_pack.yaml --name pack16  # after editing the group to g6.4xlarge

Each setting writes to its own --pred-dir, so nothing comes from the cache and
every run does the same work.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
REPO = Path(__file__).resolve().parent.parent


def run(model: str, per_gpu: int, ligands: int, cpus: int, batch: int, root: Path,
        accelerator: str = "L4") -> dict:
    """One pipeline run on one GPU, with `per_gpu` predictors sharing it."""
    tag = f"{model}_p{per_gpu}"
    flag = "--boltz-per-gpu" if model == "boltz2" else "--of3-per-gpu"
    cmd = [sys.executable, str(REPO / "aiddd_pipeline.py"),
           "--target", "ev71-2a", "--models", model, "--limit", str(ligands),
           "--max-gpus", "1", flag, str(per_gpu), "--batch-size", str(batch),
           "--cpus-per-gpu-node", str(cpus), "--accelerator", accelerator,
           "--affinity-samples", "1",   # keep Boltz-2 runs short; ranking is unaffected
           "--pred-dir", str(root / tag / "predictions"),
           "--out", str(root / tag / "results")]
    print(f"\n=== {tag}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    sys.stdout.write(p.stdout[-4000:])
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        return {"model": model, "per_gpu": per_gpu, "error": f"exit {p.returncode}"}
    util = re.search(r"GPU utilization: (\d+)% mean", p.stdout)
    done = re.search(r"Done in ([\d.]+) min", p.stdout)
    pipeline_s = float(done.group(1)) * 60 if done else wall
    return {"model": model, "per_gpu": per_gpu, "cpus_per_node": cpus, "ligands": ligands,
            "gpu_util": int(util.group(1)) if util else None,
            "pipeline_min": round(pipeline_s / 60, 1),
            "sec_per_ligand": round(pipeline_s / ligands, 1),
            "wall_min": round(wall / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus-per-gpu-node", type=int, default=8,
                    help="vCPUs on this job's GPU worker: 8 for g6.2xlarge, 16 for g6.4xlarge")
    ap.add_argument("--of3-model", default="openbind0", choices=["openbind0", "of3ft", "of3ray"],
                    help="which OpenFold3 model to sweep. of3ft is clamped to 2 per GPU by the "
                         "pipeline (6.7 GB checkpoint), so sweep openbind0 to see real packing")
    ap.add_argument("--of3-pack", default="1,2,3,4", help="OpenFold3 predictors per GPU to try")
    ap.add_argument("--boltz-pack", default="1,2", help="Boltz-2 predictors per GPU to try")
    ap.add_argument("--of3-ligands", type=int, default=64,
                    help="enough batches that every actor gets four, even at four predictors per GPU, "
                         "so startup and the tail do not dominate the high-packing settings")
    ap.add_argument("--boltz-ligands", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accelerator", default="L4",
                    help="GPU type this job's worker group provides, e.g. L40S. The pipeline "
                         "requests it by name, so a wrong value leaves every batch unschedulable")
    ap.add_argument("--out", default=None, help="where to write the summary (default: a timestamped dir)")
    args = ap.parse_args()

    root = Path(args.out or SHARED / "workshop" / "pack" /
                f"{args.accelerator.lower()}-cpu{args.cpus_per_gpu_node}-"
                f"{time.strftime('%Y%m%d-%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, packs, ligands in ((args.of3_model, args.of3_pack, args.of3_ligands),
                                  ("boltz2", args.boltz_pack, args.boltz_ligands)):
        for per_gpu in [int(x) for x in packs.split(",") if x.strip()]:
            rows.append(run(model, per_gpu, ligands, args.cpus_per_gpu_node, args.batch_size,
                            root, args.accelerator))
            (root / "summary.json").write_text(json.dumps(rows, indent=2))

    print(f"\n\n=== Packing on one {args.accelerator} with {args.cpus_per_gpu_node} vCPUs\n")
    print(f"{'model':8} {'per GPU':>7} {'GPU util':>9} {'sec/ligand':>11} {'run min':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:8} {r['per_gpu']:>7}  {r['error']}")
        else:
            print(f"{r['model']:8} {r['per_gpu']:>7} {str(r['gpu_util']) + '%':>9} "
                  f"{r['sec_per_ligand']:>11} {r['pipeline_min']:>8}")
    print(f"\nSummary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
