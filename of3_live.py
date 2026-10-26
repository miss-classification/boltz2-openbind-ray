"""OpenFold3 held in memory: load the weights once, predict many times.

`run_openfold predict` (what OF3Predictor and run_openfold_pose call) starts a process, builds
the model and reads the checkpoint on every call: 40 to 75 s on an L4, of which the model
itself is about 17 s. OF3Model does the expensive part once, in __init__, so a Ray Serve
multiplexed `load()` can return it and every later request for that model id skips it.

OF3-p2 checkpoints only load in OpenFold3 0.4.x, so this module runs in the legacy install:
give the Ray actor or task `runtime_env={"py_executable": OF3_LEGACY_PY}`.
"""

import gc
import os
import shutil
import tempfile
import time
from pathlib import Path


class OF3Model:
    """One OpenFold3 checkpoint, loaded and ready. Not thread-safe: one predict() at a time."""

    def __init__(self, ckpt_path: str, num_diffusion_samples: int = 1, data_workers: int = 2):
        # The fine-tuned checkpoint stores pathlib objects, which recent PyTorch refuses to
        # unpickle by default. Both checkpoints are trusted downloads.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        from aiddd_pipeline import OF3_CACHE
        os.environ.setdefault("OPENFOLD_CACHE", str(OF3_CACHE))
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.validator import InferenceExperimentConfig

        t0 = time.time()
        self.ckpt_path = str(ckpt_path)
        self.out = Path(tempfile.mkdtemp(prefix="of3live_"))
        config = InferenceExperimentConfig(inference_ckpt_path=Path(ckpt_path),
                                           data_module_args={"num_workers": data_workers})
        self.runner = InferenceExperimentRunner(config, num_diffusion_samples, 1, False, False, self.out)
        self.runner.setup()                     # builds the model and loads the weights into it
        # setup() loaded the EMA weights into the model itself. The separate EMA copy is only for
        # training and validation: 2.3 GB of host memory per model that inference never reads.
        self.runner.lightning_module.ema = None
        _release_memory()
        self.load_seconds = time.time() - t0

    def predict(self, queries: dict) -> Path:
        """Run {name: OpenFold3 query} and return the output folder, laid out like
        `run_openfold predict --output_dir`: <out>/<name>/seed_*/<name>_seed_*_sample_*_model.cif."""
        from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
        # The runner caches its data module per query set: drop it, keep the model.
        for name in ("data_module_config", "lightning_data_module"):
            self.runner.__dict__.pop(name, None)
        self.runner.run(InferenceQuerySet.model_validate({"seeds": [42], "queries": queries}))
        return self.out

    def __del__(self):
        """Called by Ray Serve when it evicts this model: free its memory now, not at the next GC."""
        runner = self.__dict__.pop("runner", None)
        if runner is not None:
            del runner
            _release_memory()
        shutil.rmtree(self.__dict__.get("out", ""), ignore_errors=True)


def _release_memory():
    """Collect freed tensors and hand the memory back to the OS (glibc keeps it otherwise)."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def predict_one(pred, smiles: str) -> dict:
    """One SMILES through a pipeline predictor, as a JSON-ready dict."""
    import hashlib
    import json
    import pandas as pd
    from aiddd_pipeline import validate
    lig_id = "smi_" + hashlib.sha1(smiles.encode()).hexdigest()[:12]
    batch = validate(pd.DataFrame({"ligand_id": [lig_id], "smiles": [smiles]}))
    r = pred(batch).iloc[0]
    k = pred.key
    out = {"ligand_id": lig_id, "status": r[f"{k}_status"], "error": r[f"{k}_error"],
           "metrics": json.loads(r[f"{k}_metrics"]), "seconds": float(r[f"{k}_batch_seconds"])}
    if r[f"{k}_cif"]:
        out["cif"], out["cif_path"] = Path(r[f"{k}_cif"]).read_text(), r[f"{k}_cif"]
    return out


class LiveModel:
    """What a Ray Serve multiplexed load() returns: one model id's weights, loaded, plus the pose
    cache of its version. Each answer says where it came from:
      cache    the pose was already on disk: no GPU work
      memory   computed now by weights this replica already held
      loaded   computed now, and the weights were loaded for this request (load_seconds)"""

    def __init__(self, spec: dict, target: str = "ev71-2a"):
        self.target = target
        self.load(spec)

    def load(self, spec: dict):
        """Build OpenFold3 with this spec's checkpoint, and point at its version's pose cache."""
        from aiddd_pipeline import TARGETS, OF3Predictor
        self.spec = spec
        cache = TARGETS[self.target].pred_root / f"live_of3ft_{spec['version']}"
        self.pred = OF3Predictor(key="live_of3ft", ckpt_path=spec["checkpoint"], target=self.target,
                                 diffusion_samples=spec.get("samples", 1), out_dir=str(cache))
        self.pred.model = OF3Model(spec["checkpoint"], spec.get("samples", 1))
        self.load_seconds = self.pred.model.load_seconds
        self.first = True

    @property
    def resident(self) -> bool:
        """True while the weights are loaded: False after free() or eviction."""
        pred = self.__dict__.get("pred")
        return pred is not None and pred.model is not None

    def free(self):
        """Release the weights now. Ray Serve calls __del__ (this) when it evicts the model id."""
        pred = self.__dict__.get("pred")
        if pred is not None and pred.model is not None:
            model, pred.model = pred.model, None
            model.__del__()

    __del__ = free

    def cache_files(self, smiles: str) -> list:
        import hashlib
        lig_id = "smi_" + hashlib.sha1(smiles.encode()).hexdigest()[:12]
        return [self.pred.out / f"{lig_id}.cif", self.pred.out / f"{lig_id}.json"]

    def __call__(self, smiles: str, fresh: bool = False) -> dict:
        """fresh: delete this molecule's cached pose first, so the model really runs."""
        if not self.resident:
            raise RuntimeError(f"model {self.spec['version']} was freed; load() it again")
        if fresh:
            for f in self.cache_files(smiles):
                f.unlink(missing_ok=True)
        t0 = time.time()
        out = predict_one(self.pred, smiles)
        loaded_now, self.first = self.first, False
        out["source"] = ("cache" if out["status"] == "cached" else "loaded" if loaded_now else "memory")
        out["load_seconds"] = round(self.load_seconds, 1) if loaded_now else 0.0
        out["seconds"] = round(time.time() - t0, 1)
        out["model"] = dict(self.spec)
        return out
