"""
A workshop room on the shared Service: N clients at once, each sending M prediction requests.

    python scripts/load_room.py                          # 40 clients x 6 requests
    python scripts/load_room.py --clients 10 --requests 3 --url https://...

Requests use only the cached demo compounds (aiddd_pipeline.DEMO_COMPOUNDS) and 2 model ids:
`base` and the Viewer's current route. It never asks for a fresh prediction. Prints wall time
percentiles, the error count, and counts by source (cache / memory / loaded). Run it on stage
while the room is quiet, so the Ray Serve metrics panels have traffic.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root
from aiddd_pipeline import DEMO_COMPOUNDS  # noqa: E402

DEFAULT_URL = "https://aiddd-viewer-5ms2e.cld-g54aiirwj1s8t9kt.s.anyscaleuserdata.com"


def percentile(values, q):
    values = sorted(values)
    return (
        values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]
        if values
        else float("nan")
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("AIDDD_SERVICE_URL", DEFAULT_URL))
    ap.add_argument("--clients", type=int, default=40)
    ap.add_argument("--requests", type=int, default=6, help="requests per client")
    ap.add_argument(
        "--models", nargs="*", help="model ids; default: base and the current route"
    )
    ap.add_argument("--json", help="also write the summary to this file")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    models = args.models
    if not models:
        route = requests.get(f"{url}/models", timeout=60).json()["route"]
        models = ["base"] + ([route] if route != "base" else [])
    # Every (compound, model id) pair, in turn, spread evenly over all requests
    pairs = [(cid, smi, m) for m in models for cid, smi in DEMO_COMPOUNDS.items()]
    work = [pairs[i % len(pairs)] for i in range(args.clients * args.requests)]
    print(
        f"{args.clients} clients x {args.requests} requests = {len(work)} requests to {url}, models {models}"
    )

    def client(c):
        out = []
        for cid, smi, model in work[c * args.requests : (c + 1) * args.requests]:
            t0 = time.time()
            try:
                r = requests.post(
                    f"{url}/predict", json={"smiles": smi, "model": model}, timeout=900
                )
                body = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                ok = r.status_code == 200
                out.append(
                    {
                        "compound": cid,
                        "model": model,
                        "wall": time.time() - t0,
                        "ok": ok,
                        "status": r.status_code,
                        "source": body.get("source"),
                        "zone": body.get("zone"),
                        "error": None if ok else (body.get("error") or r.text[:200]),
                    }
                )
            except Exception as e:
                out.append(
                    {
                        "compound": cid,
                        "model": model,
                        "wall": time.time() - t0,
                        "ok": False,
                        "status": None,
                        "source": None,
                        "zone": None,
                        "error": f"{type(e).__name__}: {e}"[:200],
                    }
                )
        return out

    t0 = time.time()
    with ThreadPoolExecutor(args.clients) as pool:
        results = [r for rs in pool.map(client, range(args.clients)) for r in rs]
    total = time.time() - t0

    walls = [r["wall"] for r in results]
    errors = [r for r in results if not r["ok"]]
    summary = {
        "clients": args.clients,
        "requests": len(results),
        "models": models,
        "seconds": round(total, 1),
        "p50": round(percentile(walls, 50), 2),
        "p95": round(percentile(walls, 95), 2),
        "p99": round(percentile(walls, 99), 2),
        "max": round(max(walls), 2),
        "mean": round(statistics.mean(walls), 2),
        "errors": len(errors),
        "by_source": dict(Counter(r["source"] for r in results)),
        "zones": dict(Counter(r["zone"] for r in results if r["zone"])),
    }
    print(
        f"done in {total:.0f} s: P50 {summary['p50']} s, P95 {summary['p95']} s, P99 {summary['p99']} s, "
        f"max {summary['max']} s, errors {len(errors)}"
    )
    print(
        f"by source: {summary['by_source']}"
        + (f"; zones: {summary['zones']}" if summary["zones"] else "")
    )
    for e in errors[:5]:
        print(f"  error: {e['compound']} {e['model']} HTTP {e['status']}: {e['error']}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "results": results}, indent=1)
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
