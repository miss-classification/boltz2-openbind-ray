# Handoff: finish notebook 03 (Ray Serve) for the AIDDD workshop

**For:** the Claude Code session working in `boltz2-openbind-ray`
**From:** Geoff (asleep until morning; leave everything in `RESULTS.md`, described at the end)
**Deadline:** results ready Monday morning. The workshop is Tuesday Oct 27 at 9:00. Geoff presents §6 (about 17 minutes) right after Alicia's §5.

## Context

- **Audience:** biotech and pharma scientists: medicinal chemists, computational chemists, AI team leads. They know the chemistry and do not know Ray. Several attendees are from Isomorphic Labs, and the CEO of Rowan is registered.
- **Repo state at handoff:** branch `geoff/serve-multiplex-live`, HEAD `d5e1c3d` ("Multiplexed pool: in-place updates keep loaded models"). Real multiplexing is done:
  - `of3_live.py`
  - `OF3Pool` in `serve_viewer.py`
  - `LIVE_MODELS` in `aiddd_pipeline.py`
  - `jobs/demo_*.yaml`, which add a model id and repoint the Viewer's `route`
  - `scripts/check_serve.py`

  Read these before changing anything.
- **Some of this may already be done.** You may have already covered items below. If so, skip them and say so in RESULTS.
- **New file:** `docs/images/serve_composition.svg` is in the repo but untracked. It is a diagram of the Service app for 03 (see P1-2). Commit it with your changes.

## What the team asked for on Serve

| Source | Ask |
|---|---|
| Pawarit (Slack, Oct 22) | Highlight how easily we get GPUs from **all zones**. Show how easy it is to **compose pre and post-processing** with Ray Serve. |
| Pawarit (prep call) | This audience cares about **fast in-place swaps and developer iteration speed** more than online serving at scale. Do not prioritize a live canary or traffic-split chart. |
| Pawarit (1:1) | The common complaint is "we don't have enough GPUs": cross-zone GPUs, and one service shared between researchers. Real compounds are bigger than the demo ones, so show fragment versus larger-compound latency. |
| Dry run (Sam) | Open with **one thesis slide**: a claim backed by reasons A, B and C. Say why each feature matters before showing it. Less is more. |
| Dry run (Eric + others) | "Why does a researcher care about Serve?" The answer that landed: stay in Python, test ideas ad hoc, share with colleagues, no Kubernetes. |
| Alicia | A fresh prediction takes about 1–2 min, so warm the demo compounds ahead of time. Update the README if model names change. She rebuilds the workshop template from 03 once it is final. |

## Ground rules

- **Branch:** stay on `geoff/serve-multiplex-live` and make small commits. Geoff merges to main, then asks Alicia to rebuild the template.
- **Notebooks 01 and 02 are Alicia's.** Do not edit them. Report any inconsistencies in RESULTS instead.
- **No invented numbers.** Every number you put in 03, the viewer or RESULTS comes from a run you did today, or names its source.
- **Writing style:** match the notebook's voice. Plain, literal statements, short sentences and tables. No metaphors or flourishes.
- **Presenter path:** with `USE_GPUS = True`, the notebook must behave exactly as it does now.
- **The Service (`aiddd-viewer`):** you may redeploy it freely today.
  - A code change needs a full `anyscale service deploy -f jobs/service.yaml`. `--in-place` only works for user_config changes.
  - At the end, leave it running on your final commit with `demo_1` applied and the caches warm.

## Tasks, in priority order

If you run out of time, every task you finished still stands on its own.

### P0-1. Let attendees run §6 through the shared Service

**Why:** every attendee runs with `USE_GPUS = False`, which is the default. Today:
- 6.1 does `assert USE_GPUS`, so the whole room hits an AssertionError at the start of Geoff's section.
- `SERVICE_URL` is set in Setup but never used.
- The README says 03 uses `AIDDD_SERVICE_URL`.

Calling the team's shared service from your own notebook is also the "stay in Python, share with colleagues" point from the dry run.

**What:**
- **6.1:** keep the `PoseModel` code visible. Call `serve.run` only when `USE_GPUS`. Otherwise print one line saying the presenter runs this pool and attendees use the shared Service.
- **`pose()` helper:**
  - With `USE_GPUS`, keep the handle path as it is.
  - Otherwise, `POST {SERVICE_URL}/predict` with `{"smiles": ..., "model": model_id}`.
  - **Never send `fresh=True` from attendees.** `FORCE_COLD` is presenter-only; 40 cold requests on one OF3 replica would queue for about half an hour.
  - Return the same fields as today: `source`, `seconds`, `load_seconds`, `model`.
  - Render the pose from the response body (`cif` and `format`, already aligned to the crystal when one exists), not from a file path. Check which of `wh.show_analog` / `wh.show_prediction` accepts text.
- **A model id that isn't registered yet** (for example `ev71-v1` before Geoff applies `demo_2`) returns 404 from the Viewer. Print "not deployed yet: the presenter adds it during the demo", not a traceback.
- **New short cell, the room exercise:**
  - Send the 6.2 compounds through the Service for 2 model ids: `base` and the current route.
  - Show a pivot table of RMSD per compound per model id, plus source, seconds and zone (zone once P1-1 is done).
  - Use only compounds that are already cached, and at most 2 model ids (see P0-2).

**Done when:**
- Run All with `USE_GPUS = False` and `SERVICE_URL` set finishes with no errors.
- With `SERVICE_URL` unset, §6 prints how to set it and skips.
- The presenter path is unchanged.

### P0-2. Make the Service hold up when 40 people use it at once

**Why:** the room will hit the Service at the same time. Current limits, all to verify:
- **Viewer max_ongoing_requests:** `Viewer` has `num_replicas=1`. Ray Serve's default `max_ongoing_requests` is 5 (`ray/serve/_private/constants.py`; the image is Ray 2.59). The Viewer waits on GPU calls for up to about 40 s, so 5 slow requests can block everyone, cached requests included.
- **Viewer event loop:** `Viewer.answer_key` (RDKit scoring and alignment) runs synchronously inside `async def predict`, which blocks the replica's event loop.
- **OF3Pool:**
  - It has 1 replica and runs one prediction at a time (`asyncio.Lock`).
  - `MAX_RESIDENT = 2`, but 3 ids are registered.
  - A cached pose still loads its model first, because `load()` runs before the cache check.
  - Requests that mix all 3 ids therefore load and evict over and over, and every load holds the lock.

**What (measure before and after):**
- **Viewer:** raise `max_ongoing_requests` to about 64–100, and consider autoscaling to `max_replicas` 3 on CPU.
- **Scoring:** `answer_key` → `await asyncio.to_thread(...)`.
- **Decision for Geoff:** flag it in RESULTS; don't just change it. Set the Service's `MAX_RESIDENT` from 2 to 3 so all three ids stay loaded (about 1.5 GB of host memory each; confirm with `/models` `replica_memory_gb`).
  - The trade-off: the viewer no longer shows an eviction. The notebook's `PoseModel` (max 2) still does.
  - Recommendation: 3.
- **Load script:** add `scripts/load_room.py`.
  - It runs N concurrent clients (default 40) × M requests (default 6) over cached compounds and 2 model ids.
  - It prints P50/P95/P99 wall time, error count, and counts by source.
  - Geoff may run it on stage if the room is quiet, so the metrics panels have traffic.

**Done when:** `load_room.py` at 40×6 on cached compounds gives 0 errors, with before and after P99 in RESULTS.

### P0-3. Measure on the deployed Service the numbers the talk will say

**Why:** Geoff quotes these live, and several disagree with each other today. Record all of them in RESULTS.

- **RMSD:** every 6.2 compound × {`base`, `ev71-v1`, `ev71-v2`}, run through the Service. This also warms the caches for Tuesday. Update the 6.2 table if anything differs.
- **Latency per model id:**
  - cached
  - resident (model already in memory, `fresh=True`)
  - cold (load + compute)

  Today four places disagree:
  - `COLD = {"base": 40, "ev71-v1": 35, "ev71-v2": 40}` in 6.3
  - "about 5 to 10 s" for the load in the 6.1 text
  - "the model itself is about 17 s" in the `of3_live.py` docstring
  - "about 30 s" in the viewer page

  Make all four match what you measure.
- **Fresh time versus compound size:** one fragment, one late follow-on, and the largest compound in the set, each with its heavy-atom count.
- **In-place update time:** how long from `anyscale service deploy ... --in-place` until the new route answers, for `demo_1`→`demo_2` and `demo_2`→`demo_3`. Confirm through `/models` that the resident models survive.
- **Memory:** `/models` `replica_memory_gb` with 2 and with 3 models resident.
- **Zones:** which zone each GPU replica is in (P1-1).

When you're done, leave the caches warm for every 6.2 compound in all three versions, then apply `demo_1`.

### P0-4. Fix stale or conflicting numbers in 03

- **§7 Findings** says "Models that reach 88 to 92% on SARS-CoV-2 late follow-ons fall to 1 to 24% on EV-A71 2A." The sources disagree:
  - NB01 §4: Boltz-2 87% and OF3 145k 84% on SARS-CoV-2 follow-ons.
  - NB02: 145k 88% on SARS-CoV-2 late follow-ons.
  - OpenBind-0 scores 91–93% on EV-A71 follow-ons, so it does not fall.

  Rewrite the line from the results tables (`wh.glance`, `wh.by_stage`) and name the exact models and stage.
- **x7259a, base model:** 5.3 Å in 03 §6.2, but 3.7 Å in NB01 §3 (the markdown cell after the 8-compound run).
  - Find out why: diffusion samples, the OpenFold3 0.4.3 vs 0.5 install, or the batch pipeline's ranking.
  - Add one sentence to 6.2. Report it for NB01; don't edit NB01.
- **x7101a, ev71-v1:** 4.3 Å in 03 §6.2, but 1.18 Å in Alicia's Slack table from Oct 23. Same cause? Report it so Geoff can tell Alicia.

### P1-1. Show which zone each GPU replica runs in

**Why:** this is Pawarit's first serving ask. Today it is one yaml line and one table row.

**What:**
- Record each replica's availability zone in `Boltz2Live`'s model dict and in `OF3Pool.status()`.
- Show it on the viewer's `#pool` line.
- Add it to the room-cell table.
- How to read the zone:
  - First try the Ray node label `ray.io/availability-zone` via `ray.get_runtime_context().get_node_labels()`. Check that it exists in Ray 2.59 and that Anyscale sets the label.
  - Fall back to EC2 IMDSv2 `placement/availability-zone`.

**Be honest about the result.** The Service has 2 fixed L4 nodes, and both may land in one zone. Report what you see and don't force a spread. The claim for the talk is that the Service may take GPUs from any zone in the region, not that it spreads them.

### P1-2. Composition: say it and show it

**Why:** this is Pawarit's second ask. The Viewer already composes the request:
- RDKit prep
- Boltz-2 and OF3Pool in parallel through `asyncio.gather`
- RMSD scoring and alignment

03 never says so. It also undersells one point: `OF3Pool` runs OpenFold3 0.4.x in its own Python (`runtime_env={"py_executable": ...}`), next to Boltz-2 in the image's Python, in one app. That answers "is it a big lift to bring our existing models?"

**What:**
- Add `docs/images/serve_composition.svg` to 03, either in the Service part (the markdown after 6.5) or in a new 6.0. Add 3–4 sentences that point to `Viewer.predict`.
- In the table "What Ray Serve and Anyscale add", add two rows:
  - **Composition:** CPU prep and scoring around two GPU models, in one Python app; each deployment scales on its own.
  - **A Python environment per deployment:** `runtime_env`.
- If you change `MAX_RESIDENT` or add zone, update the SVG text to match: the OF3Pool lines and the return label. Keep its style.

**Done when:** the image renders in JupyterLab from the relative path.

### P1-3. What to watch while the room runs requests (metrics)

Add one markdown table after the room cell. List the Ray Serve Grafana panels and what each one shows during the exercise:
- "QPS per deployment"
- "P50/P90/P99 latency per deployment"
- "Queue size per deployment"
- "Replicas per deployment"

These names come from `ray/python/ray/dashboard/modules/metrics/dashboards/serve_dashboard_panels.py`; check that Anyscale shows the same names. Put the exact click path to them from the Anyscale Service page in RESULTS for Geoff.

### P1-4. README

- Make the `AIDDD_SERVICE_URL` row, the "80 to 90 | 03" row in Workshop format, and the sentence after it match the new 03.
- Say how attendees get `SERVICE_URL`.

### P2. If time is left

Make the viewer readable on a projector: larger text for the route, zone and history table.

## Out of scope

- Notebooks 01 and 02.
- Slides.
- Multi-region serving: it is on the roadmap, and this is an external room.
- A live canary.
- The fragment-growing Ray Core loop.
- MSA-cache routing.
- The Boltz template.

## Hand back: `~/anyscale/aiddd-serve-prep/RESULTS.md`

```markdown
# RESULTS

Commits: <first>..<last> on geoff/serve-multiplex-live
Service: <URL>, deployed from <commit>, demo_1 applied, caches warm: yes/no

## Done / skipped / not done (one line each, with why)

## Decisions for Geoff
- MAX_RESIDENT on the Service: <2 or 3, and what you recommend>
- <anything else>

## RMSD through the Service (Å)
| Compound | base | ev71-v1 | ev71-v2 | matches 6.2? |

## Latency (s)
| Model id | cached | resident | cold (load + compute) | load alone |

## Fresh time by compound size
| Compound | stage | heavy atoms | seconds |

## In-place updates
| Step | seconds until the new route answers | resident after |

## Memory: replica_memory_gb with 2 resident / 3 resident

## Zones: Boltz2Live <zone>, OF3Pool <zone>

## Load test (load_room.py 40×6)
| | P50 | P95 | P99 | errors |
| before | | | | |
| after | | | | |

## Metrics: click path from the Anyscale Service page to the Serve panels

## Inconsistencies to tell Alicia (NB01/NB02)
```