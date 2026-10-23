"""Multiplexed Serve app: one GPU serves a house model and many program specialists.

Separate from serve_viewer.py (the hot-swap dry-run app) on purpose. Same models and
scoring, different serving pattern:

  Boltz2Live   binding affinity, as before
  OF3Pool      one @serve.multiplexed deployment. It holds many OpenFold3 checkpoints
               by id and runs the one a request asks for. Ray Serve keeps up to
               max_num_models_per_replica loaded and evicts the least-used.

A request names a model by id ("model") or by program ("program", resolved through a
routing table). The house model is just one id in the pool, the default route. Adding a
new specialist version is a registry entry, not a swap: old versions stay reachable by id
for canary, rollback and audit. Every response carries the exact version that answered.

Honest note: OpenFold3 inference runs as a subprocess that loads the checkpoint from disk
each call, so this shows the multiplexing routing and API, not a warm-model speed-up. A
real warm pool needs a long-lived in-process model (a bigger change).

    serve run serve_viewer_multi:app

pip install "ray[serve]" fastapi
"""

import asyncio
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

from aiddd_pipeline import (OF3_FT_CKPT, OF3_P2_CKPT, TARGETS, OF3Predictor,
                            pose_in_crystal_frame, relocate_paths, score_pose)
# Read-only reuse of the dry-run app's pure helpers and the affinity deployment.
from serve_viewer import Boltz2Live, canonical, predict_one

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/shared_storage/aiddd"))
RESULTS = Path(os.environ.get("AIDDD_RESULTS", SHARED / "results" / "ev71-2a"))

# The model registry: id -> weights and inference settings. "house" is the general model;
# specialists are fine-tunes. A version is (checkpoint + samples), so "more diffusion
# samples" is a real, different version of the same weights. reconfigure() replaces this
# whole table in place, so the notebook adds or removes versions with no restart.
DEFAULT_MODELS = {
    "house":   {"name": "House model: OF3-p2 145k", "checkpoint": str(OF3_P2_CKPT),
                "legacy": True, "samples": 1},
    "ev71-v2": {"name": "EV-A71 specialist v2: public fine-tune", "checkpoint": str(OF3_FT_CKPT),
                "legacy": True, "samples": 1},
}
# program -> the model id its chemists get now. Promotion = repoint a route.
DEFAULT_ROUTES = {"default": "house", "ev71-2a": "ev71-v2"}

# How many checkpoints one replica keeps loaded. House plus a couple of specialist versions
# fit; a further id evicts the least-used. Small enough to show eviction in the demo.
MAX_RESIDENT = 3

api = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 0.5, "accelerator_type": "L4"},
                  autoscaling_config={"min_replicas": 1, "max_replicas": 1,
                                      "target_ongoing_requests": 1},
                  user_config={"models": DEFAULT_MODELS})
class OF3Pool:
    """One replica holds several OpenFold3 checkpoints, keyed by model id. get_model() runs
    only on a cache miss, so its load log also shows evictions (a reload appears twice)."""

    def __init__(self):
        self.t = TARGETS["ev71-2a"]
        self.load_log = []   # (model_id, unix_time): one entry per real (re)load
        try:
            self.replica = serve.get_replica_context().replica_tag
        except Exception:
            self.replica = "local"

    def reconfigure(self, cfg: dict):
        self.registry = cfg["models"]

    @serve.multiplexed(max_num_models_per_replica=MAX_RESIDENT)
    async def get_model(self, model_id: str) -> dict:
        """Build a predictor for one checkpoint. Serve calls this only on a cache miss and
        caches the result, so each entry in load_log is a real (re)load. Cheap here: the
        weights load in the subprocess, not now."""
        spec = self.registry[model_id]
        self.load_log.append((model_id, time.time()))
        pred = OF3Predictor(
            key=f"live_multi_{model_id}", ckpt_path=spec["checkpoint"],
            binary="/opt/of3-legacy/bin/run_openfold" if spec.get("legacy") else "run_openfold",
            target="ev71-2a", diffusion_samples=spec.get("samples", 1),
            out_dir=str(self.t.pred_root / f"live_multi_{model_id}"))
        return {"id": model_id, "name": spec["name"], "pred": pred}

    async def __call__(self, smiles: str) -> dict:
        model_id = serve.get_multiplexed_model_id()
        m = await self.get_model(model_id)
        out = predict_one(m["pred"], smiles)
        recent = [mid for mid, _ in self.load_log[-8:]]
        return {**out, "replica": self.replica, "loads_on_replica": recent,
                "model": {"name": m["name"], "version": model_id,
                          "checkpoint": m["pred"].ckpt_path, "samples": m["pred"].samples}}

    async def models(self) -> dict:
        """Registry and the load log, for the status page and the notebook."""
        return {"registry": {k: {kk: v[kk] for kk in ("name", "checkpoint", "samples")}
                             for k, v in self.registry.items()},
                "replica": self.replica,
                "load_log": [{"model": mid, "at": t} for mid, t in self.load_log]}


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1},
                  user_config={"routes": DEFAULT_ROUTES, "models": DEFAULT_MODELS})
@serve.ingress(api)
class ViewerMulti:
    def __init__(self, affinity: DeploymentHandle, pool: DeploymentHandle):
        self.affinity = affinity   # Boltz-2: affinity, always run
        self.pool = pool           # OpenFold3: pose, picked by model id
        self.df = relocate_paths(pd.concat([pd.read_parquet(f) for f in sorted(RESULTS.glob("*.parquet"))],
                                           ignore_index=True))
        self.known = {}            # canonical SMILES -> first crystal of that compound
        for r in self.df.dropna(subset=["smiles"]).sort_values("complex_name").itertuples():
            if (c := canonical(r.smiles)) is not None:
                self.known.setdefault(c, r)

    def reconfigure(self, cfg: dict):
        self.routes = cfg["routes"]
        self.registry = cfg["models"]

    def resolve(self, body: dict) -> str:
        """Which model id answers: an explicit "model" wins, else the "program" route, else
        the default route."""
        if body.get("model"):
            return body["model"]
        program = body.get("program", "default")
        return self.routes.get(program, self.routes["default"])

    def answer_key(self, row, cif_path: str) -> dict:
        ref_pdb = Path(row.ref_pdb)
        ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
        try:
            rmsd, _ = score_pose(cif_path, ref_pdb, ref_sdf, {})
            return {"answer_key": {"complex": row.complex_name, "lig_rmsd": float(rmsd)},
                    "cif": pose_in_crystal_frame(cif_path, ref_pdb), "format": "pdb"}
        except Exception as e:
            return {"answer_key": {"complex": row.complex_name, "lig_rmsd": None, "error": str(e)}}

    @api.post("/predict")
    async def predict(self, body: dict):
        smiles = str(body.get("smiles", "")).strip()
        if not smiles or len(smiles) > 300:
            return JSONResponse({"status": "failed", "error": "need a SMILES under 300 chars"}, 400)
        model_id = self.resolve(body)
        if model_id not in self.registry:
            return JSONResponse({"status": "failed", "error": f"unknown model {model_id!r}",
                                 "known": sorted(self.registry)}, 400)
        # Affinity and the chosen pose model run at the same time.
        aff, pose = await asyncio.gather(
            self.affinity.remote(smiles),
            self.pool.options(multiplexed_model_id=model_id).remote(smiles))
        iptm = pose["metrics"].get("ligand_iptm", pose["metrics"].get("iptm"))
        out = {**pose, "model_id": model_id, "pose_model": pose["model"]["name"], "pose_iptm": iptm,
               "metrics": aff["metrics"], "affinity_status": aff["status"],
               "seconds": max(pose["seconds"], aff["seconds"]),
               "routes": self.routes,
               "models": {"pose": pose["model"], "affinity": aff["model"]}}
        c = canonical(smiles)
        if (row := self.known.get(c) if c else None) is not None and pose.get("cif_path"):
            out.update(self.answer_key(row, pose["cif_path"]))
        return JSONResponse(out, 200 if out["status"] in ("ok", "cached") else 422)

    @api.get("/models")
    async def models(self):
        return JSONResponse({"routes": self.routes, **(await self.pool.models.remote())})

    @api.get("/", response_class=HTMLResponse)
    def index(self):
        rows = "".join(f"<tr><td>{k}</td><td>{v['name']}</td><td>{Path(v['checkpoint']).name}</td>"
                       f"<td>{v.get('samples', 1)}</td></tr>" for k, v in self.registry.items())
        routes = "".join(f"<tr><td>{p}</td><td>{m}</td></tr>" for p, m in self.routes.items())
        return (f"<!doctype html><title>Multiplexed co-folding</title>"
                f"<style>body{{font-family:system-ui;margin:24px}}td,th{{padding:4px 10px;border-bottom:1px solid #eee;text-align:left}}</style>"
                f"<h2>Model pool</h2><table><tr><th>id</th><th>name</th><th>checkpoint</th><th>samples</th></tr>{rows}</table>"
                f"<h2>Routes (program &rarr; model)</h2><table><tr><th>program</th><th>model</th></tr>{routes}</table>"
                f"<p>POST /predict {{\"smiles\": ..., \"program\"|\"model\": ...}} &middot; GET /models for the load log.</p>")


def build_app(accelerator: str = "L4"):
    """The viewer, Boltz-2 affinity and one multiplexed OpenFold3 pool. Both GPU deployments
    take half an L4, so the whole app fits on one L4: one GPU for a small shop."""
    gpu = {"num_gpus": 0.5, "accelerator_type": accelerator}
    one = {"min_replicas": 1, "max_replicas": 1, "target_ongoing_requests": 1}
    affinity = Boltz2Live.options(ray_actor_options=gpu, autoscaling_config=one).bind()
    pool = OF3Pool.options(ray_actor_options=gpu).bind()
    return ViewerMulti.bind(affinity, pool)


app = build_app()
