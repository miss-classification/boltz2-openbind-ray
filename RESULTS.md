# RESULTS

Overnight work on HANDOFF_oct26.md. Updated as tasks finish; every number comes from a run named
next to it. This copy lives on the branch for review; delete it before merging to main.

Commits: 4f6af6e..(in progress) on geoff/serve-multiplex-live
Service: https://aiddd-viewer-5ms2e.cld-g54aiirwj1s8t9kt.s.anyscaleuserdata.com, deployed from (in progress), demo_1 applied: (pending), caches warm: (pending)

## Done / skipped / not done (one line each, with why)

- Before the handoff (commits c84aaf7..a3f1eed): real multiplexing (of3_live.py, OF3Pool,
  LIVE_MODELS, jobs/demo_*.yaml, scripts/check_serve.py), verified on the workspace cluster:
  check_serve.py passes all steps, 03_serve runs end to end with USE_GPUS = True.
- P0-1 done: §6 runs for attendees through the Service. Verified Oct 25 ~22:5x: `USE_GPUS = False` with
  `AIDDD_SERVICE_URL` set (all §6 cells and the 6.7 room cell, no errors) and unset (prints how to set it, skips).
  The "not deployed yet" message for an unregistered id: verified once demo_1 is applied (pending, end of P0-3).
  Presenter path (`USE_GPUS = True`) verified 22:2x Oct 25 on the final notebook, all code cells, no errors:
  6.3 base computed live in 42 s (12 s loading the model); 6.4 v1 from cache, 9.6 s (the model load);
  6.5 v2 from cache, 46.7 s (the model load); base again 37 s (evicted, loads again).
- P0-2 done: load_room.py 40×6, P99 118.3 s → 0.24 s warm, 0 errors (table below). Concurrency fixes, scoring
  memo, MAX_RESIDENT 3, `fresh` gated by the Viewer's `allow_fresh` (off for the room).
- P0-3 in progress: RMSD table done; latency, size, in-place, memory and zones measuring now.
- P0-4 done: §7 line rewritten from `wh.by_stage`; x7259a and x7101a causes in "Inconsistencies" below.
- P1-1 done: zone per replica (`ray.io/availability-zone` node label, EC2 metadata fallback) in responses,
  `/models`, the viewer's pool line and the 6.7 table.
- P1-2 done: `docs/images/serve_composition.svg` drawn from `Viewer.predict` (the handoff's copy was not in the
  workspace) and rendered to check it; 6.6 text and 2 table rows. Not yet checked in JupyterLab itself.
- P1-3 done: panel table after 6.7, names from Ray 2.59's `serve_dashboard_panels.py` and
  `serve_deployment_dashboard_panels.py`. Click path in the Anyscale console: not verified (no browser here).
- P1-4 done: README rows and how attendees get the URL.
- P2 done: route, zone, history and status text 14 px on the viewer page (page script tested in a JS interpreter).

## Decisions for Geoff
- MAX_RESIDENT on the Service: 3 (Geoff chose 3 on Oct 25, before going to sleep). Applied; the notebook's own
  pool keeps 2 and still shows an eviction.
- **Diffusion samples: keep 1 (recommended, unchanged) or serve 5.** Everything live (notebook 03, the Service,
  NB01 §3) and the stored benchmark use 1 sample. With 5 samples and OpenFold3's top-ranked pick, the base
  model recovers a lot, and the "base is wrong on this pocket" story weakens. 14 shortlist follow-ons, within 2 Å
  (1 sample: Service path, 21:2x Oct 25; 5 samples: in-process models, 22:3x Oct 25, scratch outputs):

  | Model | 1 sample | 5 samples |
  |---|---|---|
  | base | 1 of 14 | 6 of 14 |
  | ev71-v1 (ours) | 8 of 14 | 9 of 14 |
  | ev71-v2 (public) | 14 of 14 | 14 of 14 |

  x7259a with 5 samples: base 2.08, v1 0.57, v2 0.43 Å (1 sample: 5.3, 0.96, 0.43). A warm 5-sample prediction takes
  about 31 s, 1 sample about 25 s (one L4 to itself). Switching needs new version names (the pose caches are per
  version) and re-warming. Suggested line for Q&A: "with more samples the base model closes part of the gap; the
  fine-tune is right on every one of these with a single sample."
- **v2's model load on stage: 20 to 47 s** (6.7 GB checkpoint that also carries training state; measured in
  several runs tonight). The notebook says so in 6.5. Option, not done: a weights-only copy of the checkpoint
  (about 1.5 GB) would load in about 10 s, but it is a new file next to the published one; your call.
- **Alicia's x7101a table mixes settings.** Her v1 1.18 / v2 0.80 match tonight's 5-sample runs (1.19 / 0.82); her
  base 6.42 matches 1-sample runs (6.17), not 5 samples (1.45).

## RMSD through the Service (Å)

scripts/measure_service.py, version g9yshm5m, all 18 from the pose cache (22:1x Oct 25). 1 diffusion sample.

| Compound | base | ev71-v1 | ev71-v2 | matches 6.2? |
|---|---|---|---|---|
| x7259a | 5.20 | 0.96 | 0.43 | base was 5.3 in 6.2; now 5.2 (live runs vary by about 0.2 Å) |
| x7608a | 4.77 | 0.31 | 0.53 | yes |
| x7024a | 6.07 | 1.08 | 0.83 | yes |
| x6714a | 4.73 | 0.92 | 0.38 | yes |
| x7011a | 4.78 | 1.40 | 0.53 | yes |
| x7101a | 6.17 | 4.32 | 0.78 | yes |

## Latency (s)

scripts/measure_service.py on the Service, version g9yshm5m (22:3x Oct 25). Wall time of one /predict for x7259a
(cold) or x7608a (resident, cached), through the public URL. cold = the id was just re-registered, so the request
loads the model and computes; resident = `fresh` on a loaded model; cached = the pose from the cache.

| Model id | cached | resident | cold (load + compute) | load alone |
|---|---|---|---|---|
| base | 0.2 | 25.3 | 61.7 | 22.1 |
| ev71-v1 | 0.0 | 25.0 | 40.6 | 15.0 |
| ev71-v2 | 0.0 | 25.1 | 98.0 | 72.4 |

Loads on the Service read the checkpoint from shared storage and vary run to run: on the workspace cluster
tonight base loaded in 6 to 13 s, v1 in 8 to 10 s, v2 in 19 to 47 s. v2's checkpoint is 6.7 GB (training state
included). The 4 numbers the handoff listed as conflicting now say: 6.1 "about 5 to 10 s" (load) → update to
match this table; `COLD` in 6.3; the `of3_live.py` docstring (17 s for the model itself; measured forward
pass ~16 to 17 s inside the 25 s resident time); the viewer page's "about 30 s" (resident 25 s).

## Fresh time by compound size

Same run, ev71-v2 resident, `fresh` (model already in memory):

| Compound | stage | heavy atoms | seconds |
|---|---|---|---|
| x0554b | fragment (median heavy-atom count among fragments) | 13 | 23.5 |
| x7259a | late follow-on | 28 | 24.8 |
| x7096a | largest compound in the set | 35 | (107, invalid: the OF3Pool replica restarted during this request; see below) |

Fragment to late follow-on adds about 1 s: the protein (about 150 residues) dominates the cost, not the ligand.

## In-place updates

Same run, the real jobs/demo_*.yaml, `anyscale service deploy --in-place`, timed from the command until /models
shows the new route and registry. Anyscale keeps the Service in UPDATING for 1 to 2 minutes after that; a second
deploy is refused until it is RUNNING again.

| Step | seconds until the new route answers | resident after |
|---|---|---|
| demo_1 → demo_2 | 10.2 | [] (nothing was loaded; survival not tested in this run, re-test pending) |
| demo_2 → demo_3 | 12.2 | [] |

## Memory: replica_memory_gb with 2 resident / 3 resident

Workspace cluster, check_serve.py on the final server code (22:1x Oct 25): base only 7.0 GB, 2 resident
10.1 GB, 3 resident 13.3 GB (`/models` replica_memory_gb, the OF3Pool replica's resident set size).
Service (measure_service.py, 22:3x): 0 resident 0.2 GB, 1 resident 8.5 GB, 2 resident 11.5 GB, 3 resident
14.6 GB. The L4 node has 32 GB and may also host a Boltz2Live replica.

**Open issue (investigating):** in that run the OF3Pool replica restarted during the largest compound's fresh
prediction with 3 models resident: afterwards nothing was resident and the replica had moved from us-west-2a to
us-west-2c. Likely the node's memory (Ray's memory monitor kills the largest worker). Service logs need an API
key this workspace does not have, so it is being reproduced on the workspace cluster.

## Zones

Workspace cluster (check_serve.py, 22:1x Oct 25): OF3Pool in us-west-2a and Boltz2Live in us-west-2b, so one
app took GPUs from 2 zones. An earlier run on the same cluster had both in us-west-2b. Service: pending.

## Load test (load_room.py 40×6)

240 requests per run over the 6 cached demo compounds and model ids base and ev71-v2. The before
tail is queueing, not compute: every answer came from the cache.

| | P50 (s) | P95 (s) | P99 (s) | errors |
|---|---|---|---|---|
| before: Service version ibhuigyf (commit 4f6af6e: Viewer 1 replica, max_ongoing_requests 5; OF3Pool loads the model before the cache check; Boltz2Live sync), 21:5x Oct 25 | 10.39 | 114.57 | 118.25 | 0 |
| after 1: version ya7hppeb (commit ddb0700: Viewer 1 to 3 replicas, max_ongoing_requests 100, scoring in a thread; OF3Pool and Boltz2Live answer cached poses with no model or lock; MAX_RESIDENT 3), 22:2x Oct 25 | 10.64 | 12.23 | 12.42 | 0 |
| after 2, first burst: version g9yshm5m (commit d06d552: scoring memoized per pose file; Viewer starts at 2 replicas), 22:4x Oct 25 | 2.05 | 4.36 | 5.14 | 0 |
| after 2, second burst (scores already memoized on both Viewer replicas) | 0.16 | 0.20 | 0.24 | 0 |

After 1 removed the tail (P99 118 s to 12 s). Its P50 of 10.6 s was the Viewer's CPU: 240 requests in 65 s
is 3.7 requests per second, about 0.25 s of RDKit scoring and alignment per request on 1 replica, repeated
for the same 12 poses. After 2 scores each pose once per replica and starts with 2 replicas: the burst of 240
took 166 s before, 65 s after 1, 15 s after 2 (first burst) and about 1 s once warm. Running load_room.py once
before the session leaves the Service in the warm state.

## Metrics: click path from the Anyscale Service page to the Serve panels

## Inconsistencies to tell Alicia (NB01/NB02)

- **The 145k base model (`of3p2`) has no stored results in this workspace**: `/mnt/shared_storage/aiddd/results_base-of3p2/`
  is empty, so `wh.load_results` has no `of3p2_*` columns. NB01's "OpenFold3 145k 84%" on SARS-CoV-2 follow-ons and
  "36%" on EV-A71 2A, and NB02's "145k 88% / 24%" table, cannot be checked against stored tables here.
- **x7259a, base model: NB01 §3 says 3.7 Å; every run tonight gives about 5 Å.** Same predictor class
  (`OF3Predictor`, legacy OpenFold3 0.4.3, 1 diffusion sample, seed 42), scratch outputs (tested 22:0x Oct 25):

  | Run | RMSD (Å) |
  |---|---|
  | x7259a alone, run 1 | 5.25 |
  | x7259a alone, run 2 | 5.05 |
  | x7259a in NB01 §3's batch of 2 (with x7025a) | 5.25 |
  | Live pool / Service (in-process model), 3 runs | 5.20, 5.20, 5.33 |

  Batch composition does not explain it; run to run varies by about 0.2 Å. With 5 diffusion samples and
  OpenFold3's top-ranked pick (`OF3Predictor(diffusion_samples=5)`, 22:1x Oct 25), x7259a base is 2.07 Å: still
  not 3.7 Å. NB01's 3.7 Å is unexplained; it may come from an earlier code state (for example a run before
  `OF3Predictor` named the MSA file so OpenFold3 reads it).
- **x7101a, ev71-v1 (`ft-p2-evfrag-1000s-8g-1023-0139`): Alicia's Slack table says 1.18 Å; tonight 4.3 to 5.5 Å.**

  | Run | RMSD (Å) |
  |---|---|
  | alone, 1 sample | 5.04 |
  | in a batch of 8 late follow-ons, 1 sample | 5.47 |
  | `run_openfold_pose`, 1 sample (Oct 25 evening) | 5.05 |
  | Service path (benchmark of 13 follow-ons, Oct 25 evening) | 4.32 |

  **Cause found: diffusion samples.** With 5 samples and OpenFold3's top-ranked pick (22:1x Oct 25):
  ev71-v1 **1.19 Å** (Alicia: 1.18) and ev71-v2 **0.82 Å** (Alicia: 0.80). Her table almost certainly used 5
  samples; every live path in this repo (notebook 03, the Service, NB01 §3) uses 1. Still worth asking: which run (`-0139` or `-0140`, the example in `workshop_helpers.OURS_RUN`'s comment), how many
  diffusion samples, and which scoring script produced 1.18 Å.
