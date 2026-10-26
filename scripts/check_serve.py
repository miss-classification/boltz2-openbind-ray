"""
Check the Ray Serve app end to end: deploy it, load the leaderboard, then walk the model
lifecycle the live demo shows and check what multiplexing does at each step:

  1. base model, first request          weights load on this request            source=loaded
  2. base model, another molecule       weights already in memory               source=memory
  3. the same molecule again            pose from the cache, no GPU work        source=cache
  4. our fine-tune (ev71-v1)            a second model id loads beside the first
  5. in-place update: add ev71-v2 and route the program to it (reconfigure, no restart:
     the loaded models survive it)
  6. ev71-v2                            a third id evicts the least recently used one

    python scripts/check_serve.py                       # deploy, check, delete the app
    python scripts/check_serve.py --keep                # leave the app running afterwards

Deploys as its own application ("aiddd-check" at /check), so other Serve apps on the
cluster, such as notebook 03's "aiddd-pose", keep running. Deploys declaratively, through
Serve's REST API: the path `serve deploy` and `anyscale service deploy --in-place` take, where
a change to user_config only is a lightweight update (reconfigure(), no restart). serve.run
would start new replicas on every call.

Needs L4 GPUs for the Boltz-2 and OpenFold3 replicas. "fresh" requests delete the molecule's
cached pose first, so the model really runs.
"""

import argparse
import sys
import time
from pathlib import Path

import ray
import requests
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root
from aiddd_pipeline import LIVE_MODELS  # noqa: E402

APP, PREFIX = "aiddd-check", "/check"
URL = "http://localhost:8000" + PREFIX
DASHBOARD = "http://localhost:8265"
X7259A = "CSc1ncnc(N2CCC[C@@H]2C2CC2)c1CC(=O)N1CCC[C@H](C(N)=O)C1"
X7608A = "CSc1ncnc(N(C)CC(F)(F)F)c1CC(=O)N1C[C@@H](C(N)=O)C[C@@H]2C[C@@H]21"


def deploy(model_ids, route, timeout=1800):
    """Deploy serve_viewer:app from a config with this registry and route, and wait until the
    Viewer serves that route. On a running app this changes only user_config."""
    models = {i: LIVE_MODELS[i] for i in model_ids}
    # The code this driver uploaded (a gcs:// package) and its pip packages, as the app's runtime_env
    env = {k: v for k, v in ray.get_runtime_context().runtime_env.items() if k in ("working_dir", "pip", "env_vars")}
    ServeSubmissionClient(DASHBOARD).deploy_applications({"applications": [{
        "name": APP, "route_prefix": PREFIX, "import_path": "serve_viewer:app", "runtime_env": env,
        "deployments": [{"name": "OF3Pool", "user_config": {"models": models}},
                        {"name": "Viewer", "user_config": {"route": route}}]}]})
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = requests.get(f"{URL}/models", timeout=30).json()
            if st.get("route") == route and sorted(st["registry"]) == sorted(model_ids):
                return st
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"app not serving route {route} with {model_ids} after {timeout} s")


def predict(smiles, model=None, fresh=False):
    t0 = time.time()
    r = requests.post(f"{URL}/predict", json={"smiles": smiles, "model": model, "fresh": fresh}, timeout=900)
    body = r.json()
    assert r.status_code == 200, body
    assert body["cif"].startswith(("data_", "HEADER", "ATOM", "REMARK", "CRYST1", "MODEL")), body["cif"][:80]
    body["wall"] = time.time() - t0
    return body


def show(step, b):
    rmsd = (b.get("answer_key") or {}).get("lig_rmsd")
    load = f" + {b['load_seconds']:.0f} s load" if b["load_seconds"] else ""
    print(f"[ok] {step:<34} {b['model_id']:<8} source={b['source']:<7} {b['wall']:5.1f} s wall{load}"
          f"  RMSD {rmsd if rmsd is None else round(rmsd, 2)}  resident={b['resident']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the app running after the checks")
    args = ap.parse_args()

    ray.init(runtime_env={"working_dir": str(Path(__file__).resolve().parents[1]),
                          "excludes": ["image", "docs", ".git", "*.ipynb", "__pycache__"]},
             ignore_reinit_error=True, logging_level="ERROR")
    t0 = time.time()
    deploy(["base", "ev71-v1"], route="base")
    print(f"[ok] app deployed in {time.time() - t0:.0f} s (includes GPU node startup)")

    page = requests.get(URL, timeout=60)
    assert page.status_code == 200 and "<table>" in page.text, page.text[:300]
    print(f"[ok] leaderboard page: {page.text.count('data-cid=')} complexes listed")

    b = predict(X7259A, fresh=True)
    show("1 base, first request", b)
    assert b["model_id"] == "base" and b["source"] == "loaded", b["source"]
    b = predict(X7608A, fresh=True)
    show("2 base, model in memory", b)
    assert b["source"] == "memory", b["source"]
    b = predict(X7608A)
    show("3 base, same molecule again", b)
    assert b["source"] == "cache" and b["wall"] < 10, (b["source"], b["wall"])
    b = predict(X7259A, model="ev71-v1", fresh=True)
    show("4 ev71-v1, a second model id", b)
    assert b["source"] == "loaded" and sorted(b["resident"]) == ["base", "ev71-v1"], b["resident"]

    t0 = time.time()
    status = deploy(["base", "ev71-v1", "ev71-v2"], route="ev71-v2")
    print(f"[ok] {'5 in-place update':<34} {time.time() - t0:.0f} s; route={status['route']} "
          f"resident={status['resident']} memory={status['replica_memory_gb']} GB")
    assert status["route"] == "ev71-v2" and "ev71-v2" in status["registry"], status
    assert sorted(status["resident"]) == ["base", "ev71-v1"], "the update restarted the pool replica"

    b = predict(X7259A, fresh=True)
    show("6 ev71-v2 by route, evicts the LRU", b)
    assert b["model_id"] == "ev71-v2" and b["source"] == "loaded", b
    assert sorted(b["resident"]) == ["ev71-v1", "ev71-v2"], b["resident"]
    status = requests.get(f"{URL}/models", timeout=60).json()
    print(f"[ok] pool after eviction: resident={status['resident']} memory={status['replica_memory_gb']} GB")

    status = deploy(["base"], route="base")
    print(f"[ok] {'7 reset to step 1 (base only)':<34} resident={status['resident']} memory={status['replica_memory_gb']} GB")
    assert status["resident"] == [], "ids that left the registry should be freed"
    gone = requests.post(f"{URL}/predict", json={"smiles": X7259A, "model": "ev71-v2"}, timeout=900)
    assert gone.status_code == 404 and "unknown model id" in gone.text, (gone.status_code, gone.text[:300])
    print(f"[ok] {'  ev71-v2 by id after the reset':<34} rejected: not registered")
    b = predict(X7259A)
    show("8 base after the reset", b)
    assert b["model_id"] == "base" and b["resident"] == ["base"], b["resident"]

    bad = requests.post(f"{URL}/predict", json={"smiles": "not-a-molecule"}, timeout=900)
    assert bad.status_code in (400, 422), bad.status_code
    print(f"[ok] invalid SMILES rejected with HTTP {bad.status_code}")

    print("\nAll checks passed.")
    if args.keep:
        print(f"App is running at {URL}. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)
    serve.delete(APP)


if __name__ == "__main__":
    main()
