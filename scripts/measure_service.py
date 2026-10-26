"""
Measure on the deployed Service the numbers notebook 03 and the talk quote, then leave it on
demo step 1. Every step is an in-place update (user_config only), so nothing restarts.

    python scripts/measure_service.py                    # all measurements, then apply demo_1
    python scripts/measure_service.py --json out.json

1. RMSD: every demo compound x every model id, through the Service (cached poses)
2. Latency per model id: cached, resident (fresh, model in memory), cold (model loaded for the request)
3. Fresh time by compound size: a fragment, a late follow-on, the largest compound in the set
4. Memory: /models replica_memory_gb with 2 and 3 models resident
5. In-place updates: seconds from `anyscale service deploy --in-place` until the new route answers,
   for demo_1 -> demo_2 and demo_2 -> demo_3, and which models stay resident
6. Zones of the GPU replicas

Fresh requests need the Viewer's allow_fresh, which this script turns on with an in-place update and
off again at the end (demo_1 leaves it off).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aiddd_pipeline import DEMO_COMPOUNDS  # noqa: E402

NAME = "aiddd-viewer"
DEFAULT_URL = "https://aiddd-viewer-5ms2e.cld-g54aiirwj1s8t9kt.s.anyscaleuserdata.com"
IDS = ["base", "ev71-v1", "ev71-v2"]


def demo_config(step):
    return yaml.safe_load((ROOT / "jobs" / f"demo_{step}.yaml").read_text())


def with_changes(cfg, ids=None, allow_fresh=None):
    """A copy of a demo config with a different registry or allow_fresh."""
    cfg = copy.deepcopy(cfg)
    for d in cfg["applications"][0]["deployments"]:
        if d["name"] == "OF3Pool" and ids is not None:
            full = demo_config("3_next_ft")["applications"][0]["deployments"][0][
                "user_config"
            ]["models"]
            d["user_config"]["models"] = {i: full[i] for i in ids}
        if d["name"] == "Viewer" and allow_fresh is not None:
            d["user_config"]["allow_fresh"] = allow_fresh
    return cfg


class Service:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def models(self):
        return requests.get(f"{self.url}/models", timeout=60).json()

    def predict(self, smiles, model, fresh=False):
        t0 = time.time()
        r = requests.post(
            f"{self.url}/predict",
            json={"smiles": smiles, "model": model, "fresh": fresh},
            timeout=900,
        )
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"{model} HTTP {r.status_code}: {body.get('error')}")
        body["wall"] = round(time.time() - t0, 1)
        return body

    @staticmethod
    def wait_running(timeout=900):
        import anyscale

        t0 = time.time()
        while time.time() - t0 < timeout:
            if str(anyscale.service.status(name=NAME).state).endswith("RUNNING"):
                return
            time.sleep(3)
        raise TimeoutError(f"{NAME} not RUNNING after {timeout} s")

    def apply(self, cfg, timeout=900):
        """In-place update; seconds from the command until /models shows the new route and registry."""
        route = next(
            d for d in cfg["applications"][0]["deployments"] if d["name"] == "Viewer"
        )["user_config"]["route"]
        ids = sorted(
            next(
                d
                for d in cfg["applications"][0]["deployments"]
                if d["name"] == "OF3Pool"
            )["user_config"]["models"]
        )
        fresh = next(
            d for d in cfg["applications"][0]["deployments"] if d["name"] == "Viewer"
        )["user_config"].get("allow_fresh", False)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        try:
            self.wait_running()  # Anyscale refuses a deploy while the previous update is still in progress
            t0 = time.time()
            r = subprocess.run(
                ["anyscale", "service", "deploy", "-f", f.name, "--in-place"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if r.returncode:
                raise RuntimeError(
                    f"anyscale service deploy failed: {(r.stderr or r.stdout)[-800:]}"
                )
            while time.time() - t0 < timeout:
                try:
                    m = self.models()
                    if (
                        m["route"] == route
                        and sorted(m["registry"]) == ids
                        and m.get("allow_fresh") == fresh
                    ):
                        return round(time.time() - t0, 1), m
                except Exception:
                    pass
                time.sleep(2)
        finally:
            os.unlink(f.name)
        raise TimeoutError(f"route {route} with {ids} not live after {timeout} s")


def heavy_atoms(smiles):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("AIDDD_SERVICE_URL", DEFAULT_URL))
    ap.add_argument("--json", help="write every measurement to this file")
    args = ap.parse_args()
    s, out = Service(args.url), {}
    full = demo_config("3_next_ft")
    log = lambda *a: print(*a, flush=True)

    secs, m = s.apply(with_changes(full, allow_fresh=True))
    log(f"setup: all 3 ids registered, allow_fresh on ({secs} s); zone {m.get('zone')}")

    # 1. RMSD for every compound and model id (cached)
    out["rmsd"] = {}
    for cid, smi in DEMO_COMPOUNDS.items():
        for mid in IDS:
            b = s.predict(smi, mid)
            out["rmsd"].setdefault(cid, {})[mid] = {
                "rmsd": (b.get("answer_key") or {}).get("lig_rmsd"),
                "source": b["source"],
                "wall": b["wall"],
            }
        log(
            "rmsd",
            cid,
            {
                k: (round(v["rmsd"], 2) if v["rmsd"] is not None else None, v["source"])
                for k, v in out["rmsd"][cid].items()
            },
        )

    # 2 and 4. Latency per id; memory with 2 and 3 resident
    out["latency"], out["memory"] = {}, {}
    x = DEMO_COMPOUNDS["x7259a"]
    y = DEMO_COMPOUNDS["x7608a"]
    for mid in IDS:
        others = [i for i in IDS if i != mid]
        _, m = s.apply(with_changes(full, ids=others, allow_fresh=True))  # frees mid
        out["memory"][f"{len(m['resident'])} resident {m['resident']}"] = m[
            "replica_memory_gb"
        ]
        _, m = s.apply(
            with_changes(full, ids=IDS, allow_fresh=True)
        )  # registers it again, not loaded
        cold = s.predict(x, mid, fresh=True)
        resident = s.predict(y, mid, fresh=True)
        cached = s.predict(y, mid)
        out["latency"][mid] = {
            "cold": cold["wall"],
            "load": cold["load_seconds"],
            "cold_source": cold["source"],
            "resident": resident["wall"],
            "resident_source": resident["source"],
            "cached": cached["wall"],
            "cached_source": cached["source"],
        }
        log("latency", mid, out["latency"][mid])
    m = s.models()
    out["memory"][f"{len(m['resident'])} resident {m['resident']}"] = m[
        "replica_memory_gb"
    ]
    log("memory", out["memory"])

    # 3. Fresh time by compound size, on a resident model
    import workshop_helpers as wh

    ev = wh.load_results("ev71-2a").dropna(subset=["smiles"]).drop_duplicates("smiles")
    ev["heavy"] = ev.smiles.map(heavy_atoms)
    frag = (
        ev[ev.fragment_screen.astype(bool)]
        .sort_values("heavy")
        .iloc[len(ev[ev.fragment_screen.astype(bool)]) // 2]
    )
    big = ev.sort_values("heavy").iloc[-1]
    out["size"] = []
    for label, cid, smi in (
        ("fragment (median size)", frag.complex_name, frag.smiles),
        ("late follow-on (x7259a)", "A71EV2A-x7259a", x),
        ("largest compound in the set", big.complex_name, big.smiles),
    ):
        s.predict(smi, "ev71-v2")  # model resident, Boltz-2 cached
        b = s.predict(smi, "ev71-v2", fresh=True)
        out["size"].append(
            {
                "compound": cid,
                "label": label,
                "heavy_atoms": heavy_atoms(smi),
                "wall": b["wall"],
                "pose_seconds": b["seconds"],
                "source": b["source"],
            }
        )
        log("size", out["size"][-1])

    # 3b. Both GPU models live at once (a compound no cache has): the OF3Pool replica must survive
    second = ev.sort_values("heavy").iloc[-2]
    before = s.models()
    b = s.predict(second.smiles, "ev71-v2", fresh=True)
    after = s.models()
    out["both_live"] = {
        "compound": second.complex_name,
        "heavy_atoms": int(second.heavy),
        "wall": b["wall"],
        "source": b["source"],
        "resident_before": before["resident"],
        "resident_after": after["resident"],
        "zone_before": before["zone"],
        "zone_after": after["zone"],
        "survived": after["resident"] == before["resident"]
        and after["zone"] == before["zone"],
    }
    log("both live", out["both_live"])

    # 5. The demo's in-place updates, with the real configs. Load base first (a fresh request), so the
    # steps show whether a loaded model survives them.
    out["inplace"] = []
    secs, m = s.apply(with_changes(demo_config("1_base"), allow_fresh=True))
    s.predict(x, "base", fresh=True)
    secs, m = s.apply(demo_config("1_base"))
    log(f"demo_1 applied ({secs} s), resident {m['resident']}")
    for step in ("2_ft", "3_next_ft"):
        before = s.models()["resident"]
        secs, m = s.apply(demo_config(step))
        out["inplace"].append(
            {
                "step": step,
                "seconds": secs,
                "resident_before": before,
                "resident_after": m["resident"],
            }
        )
        log("inplace", out["inplace"][-1])
        s.predict(x, m["route"])

    # 6. Zones
    b = s.predict(x, "base")
    out["zones"] = {"OF3Pool": b.get("zone"), "Boltz2Live": b.get("affinity_zone")}
    log("zones", out["zones"])

    secs, m = s.apply(demo_config("1_base"))
    log(
        f"final: demo_1 applied ({secs} s); route {m['route']}, resident {m['resident']}, allow_fresh {m.get('allow_fresh')}"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
