# RESULTS

Overnight work on HANDOFF_oct26.md. Updated as tasks finish; every number comes from a run named
next to it. This copy lives on the branch for review; delete it before merging to main.

Commits: 4f6af6e..(in progress) on geoff/serve-multiplex-live
Service: https://aiddd-viewer-5ms2e.cld-g54aiirwj1s8t9kt.s.anyscaleuserdata.com, deployed from (in progress), demo_1 applied: (pending), caches warm: (pending)

## Done / skipped / not done (one line each, with why)

- Before the handoff (commits c84aaf7..a3f1eed): real multiplexing (of3_live.py, OF3Pool,
  LIVE_MODELS, jobs/demo_*.yaml, scripts/check_serve.py), verified on the workspace cluster:
  check_serve.py passes all steps, 03_serve runs end to end with USE_GPUS = True.
- P0-1: in progress
- P0-2: in progress
- P0-3: pending
- P0-4: pending
- P1-1: pending
- P1-2: pending. docs/images/serve_composition.svg was not in the workspace (not on disk, not in git).
- P1-3: pending
- P1-4: pending
- P2: pending

## Decisions for Geoff
- MAX_RESIDENT on the Service: 3 (Geoff chose 3 on Oct 25, before going to sleep).

## RMSD through the Service (Å)
| Compound | base | ev71-v1 | ev71-v2 | matches 6.2? |
|---|---|---|---|---|

## Latency (s)
| Model id | cached | resident | cold (load + compute) | load alone |
|---|---|---|---|---|

## Fresh time by compound size
| Compound | stage | heavy atoms | seconds |
|---|---|---|---|

## In-place updates
| Step | seconds until the new route answers | resident after |
|---|---|---|

## Memory: replica_memory_gb with 2 resident / 3 resident

Workspace cluster, check_serve.py on the final server code (22:1x Oct 25): base only 7.0 GB, 2 resident
10.1 GB, 3 resident 13.3 GB (`/models` replica_memory_gb, the OF3Pool replica's resident set size). The L4
node has 32 GB. Service numbers: pending.

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
| after 2: version g9yshm5m (scoring memoized per pose file; Viewer starts at 2 replicas) | pending | | | |

After 1 removed the tail (P99 118 s to 12 s). Its P50 of 10.6 s was the Viewer's CPU: 240 requests in 65 s
is 3.7 requests per second, about 0.25 s of RDKit scoring and alignment per request on 1 replica, repeated
for the same 12 poses. After 2 scores each pose once per replica and starts with 2 replicas.

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
