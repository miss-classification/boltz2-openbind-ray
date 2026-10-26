# RESULTS

Overnight work on HANDOFF_oct26.md (night of Oct 25 to 26). Every number comes from a run named next to it.
This copy lives on the branch for review; delete it before merging to main.

Commits: 4f6af6e..(this file's commit) on geoff/serve-multiplex-live
Service: https://aiddd-viewer-5ms2e.cld-g54aiirwj1s8t9kt.s.anyscaleuserdata.com, deployed from cbe96b1 (version
bsqwk14c), demo_1 applied: yes (route `base`, `allow_fresh` off), caches warm: yes (all 18 demo poses answer from
the pose cache; both Viewer replicas have scored them). Later commits change only notebooks, Markdown and
scripts/, which the Service does not upload.

## Before Tuesday morning

1. `python scripts/load_room.py` once: confirms the Service answers and leaves the Viewer replicas warm (last run:
   0 errors, P99 0.32 s).
2. Leave `demo_1` applied. On stage: `demo_2`, then `demo_3`, each `--in-place` (10 to 16 s each, measured).
3. Edit no `.py` file between the deploy and the demo steps. Notebooks, Markdown, `scripts/` and `jobs/` are
   excluded from the upload, so editing them is safe.
4. Share the Service URL (above) with the room as `AIDDD_SERVICE_URL` (README, Workshop format).

## Done / skipped / not done

- Before the handoff (c84aaf7..a3f1eed): real multiplexing (of3_live.py, OF3Pool, LIVE_MODELS, jobs/demo_*.yaml,
  scripts/check_serve.py).
- **P0-1 done.** §6 runs for attendees through the Service. Verified with `USE_GPUS = False`:
  `AIDDD_SERVICE_URL` set (all §6 cells and the 6.7 room cell, no errors), unset (each cell prints how to set it
  and skips), and with demo_1 applied (6.4 and 6.5 print "ev71-v1 is not deployed yet: the presenter adds it
  during the demo"). Presenter path (`USE_GPUS = True`) verified on the final notebook, all code cells, no
  errors. Requests report the wall time the attendee waited; the last cell stops the pool only on the
  presenter path.
- **P0-2 done.** load_room.py 40×6: P99 118.3 s before, 0.24 to 0.32 s warm after, 0 errors (table below).
  Two more problems found and fixed on the way: a live Boltz-2 prediction blocked its replica's event loop (Ray
  2.59 runs sync methods on the loop), and OF3Pool restarted when a live Boltz-2 prediction shared its node
  (memory; see Memory).
- **P0-3 done.** All measurements below, on the final version. 4 conflicting statements aligned: 6.1 load time,
  `COLD` in 6.3, the of3_live.py docstring, the viewer page.
- **P0-4 done.** §7 line rewritten from `wh.by_stage`. x7259a and x7101a gaps explained or narrowed (Inconsistencies).
- **P1-1 done.** Zone per replica (`ray.io/availability-zone` node label, EC2 metadata fallback) in every response,
  `/models`, the viewer's pool line and the 6.7 table.
- **P1-2 done.** `docs/images/serve_composition.svg`: the handoff's copy was not in the workspace, so it is drawn
  from `Viewer.predict` and rendered to check it; nbconvert resolves `docs/images/serve_composition.svg` from the
  notebook. Not opened in JupyterLab itself (no browser here). 6.6 text and the 2 table rows added.
- **P1-3 done, click path not verified.** Panel table after 6.7; panel names from Ray 2.59's
  `serve_dashboard_panels.py` and `serve_deployment_dashboard_panels.py`.
- **P1-4 done.** README: the `AIDDD_SERVICE_URL` row, the 03 row in Workshop format, how attendees get the URL.
- **P2 done.** Route, zone, history and status text at 14 px on the viewer page.
- Page script tests: no browser here, so the viewer page's script ran in a JavaScript interpreter (quickjs) with
  stubbed browser APIs: `predict()`, the result line, the history table, the pool line, the run-live box.

## Decisions for Geoff

- **MAX_RESIDENT on the Service: 3** (you chose 3). Applied. The notebook's own pool keeps 2 and still shows an
  eviction.
- **Diffusion samples: keep 1 (recommended, unchanged) or serve 5.** Everything live (notebook 03, the Service,
  NB01 §3) and the stored benchmark use 1 sample. With 5 samples and OpenFold3's top-ranked pick, the base model
  recovers a lot. 14 shortlist follow-ons within 2 Å (1 sample: Service path; 5 samples: in-process models,
  scratch outputs):

  | Model | 1 sample | 5 samples |
  |---|---|---|
  | base | 1 of 14 | 6 of 14 |
  | ev71-v1 (ours) | 8 of 14 | 9 of 14 |
  | ev71-v2 (public) | 14 of 14 | 14 of 14 |

  x7259a with 5 samples: base 2.08, v1 0.57, v2 0.43 Å (1 sample: 5.2, 0.96, 0.43). A warm prediction takes about
  31 s with 5 samples and 25 s with 1. Switching needs new version names (the pose caches are per version) and
  re-warming. A line for Q&A: "with more samples the base model closes part of the gap; the public fine-tune is
  right on all 14 with a single sample."
- **v2's model load: 20 to 74 s** (6.7 GB checkpoint that also carries training state). On stage it matters
  only on the presenter's pool (6.5); the Service answers v2's demo poses from the cache without loading it.
  Option, not done: a weights-only copy of the checkpoint (about 1.5 GB) would load several times faster, but it
  is a new file next to the published one.
- **Alicia's x7101a table mixes settings.** Her v1 1.18 / v2 0.80 match tonight's 5-sample runs (1.19 / 0.82); her
  base 6.42 matches 1-sample runs (6.17), not 5 samples (1.45).

## RMSD through the Service (Å)

scripts/measure_service.py, final version bsqwk14c, all 18 from the pose cache. 1 diffusion sample.

| Compound | base | ev71-v1 | ev71-v2 | matches 6.2? |
|---|---|---|---|---|
| x7259a | 5.20 | 0.96 | 0.43 | yes after the update: 6.2 said 5.3 for base, now 5.2 (live runs vary by about 0.2 Å) |
| x7608a | 4.77 | 0.31 | 0.53 | yes |
| x7024a | 6.07 | 1.08 | 0.83 | yes |
| x6714a | 4.73 | 0.92 | 0.38 | yes |
| x7011a | 4.78 | 1.40 | 0.53 | yes |
| x7101a | 6.17 | 4.32 | 0.78 | yes |

## Latency (s)

Same run. Wall time of one /predict through the public URL. cold = the id was just re-registered, so the request
loads the model and computes; resident = `fresh` on a loaded model; cached = the pose from the cache.

| Model id | cached | resident | cold (load + compute) | load alone |
|---|---|---|---|---|
| base | 0.0 | 25.2 | 62.0 | 22.6 |
| ev71-v1 | 0.0 | 25.1 | 41.1 | 15.4 |
| ev71-v2 | 0.0 | 25.1 | 99.9 | 74.3 |

The earlier run on version g9yshm5m gave the same within 2.5 s. Loads read the checkpoint from shared storage; on
the workspace cluster they took 6 to 13 s (base), 8 to 10 s (v1) and 19 to 47 s (v2). Inside the 25 s resident
time the model's forward pass is about 17 s (OpenFold3's progress bar).

## Fresh time by compound size

Same run, ev71-v2 in memory, `fresh`:

| Compound | stage | heavy atoms | seconds |
|---|---|---|---|
| x0554b | fragment (median heavy-atom count among fragments) | 13 | 23.7 |
| x7259a | late follow-on | 28 | 25.0 |
| x7096a | largest compound in the set | 35 | 25.6 |

The protein (about 150 residues) dominates the cost: a compound 3 times larger adds 2 s. Both models live at
once on a new 34-heavy-atom compound (x7190a: OpenFold3 and Boltz-2 affinity, nothing cached): 138.5 s, almost
all Boltz-2, and the OF3Pool replica kept its 3 loaded models and its node.

## In-place updates

Same run: the real jobs/demo_*.yaml with `anyscale service deploy --in-place`, timed from the command until
/models shows the new route and registry. base was loaded before the steps.

| Step | seconds until the new route answers | resident after |
|---|---|---|
| demo_1 → demo_2 | 15.9 | ['base'] (survived) |
| demo_2 → demo_3 | 10.0 | ['base'] (survived) |

Anyscale keeps the Service in UPDATING for 1 to 2 minutes after the route answers, and refuses another deploy
until it is RUNNING again: leave a minute or two between demo steps.

## Memory: replica_memory_gb with 2 resident / 3 resident

Same run (OF3Pool replica's resident set size, `/models`): 0 resident 0.2 GB, 1 resident 8.5 GB, **2 resident
11.5 GB, 3 resident 14.6 GB**. On the workspace cluster: 7.0, 10.1, 13.3 GB for 1, 2, 3.

**Found and fixed: OF3Pool and a live Boltz-2 prediction must not share a node.** In an earlier
measure_service.py run the OF3Pool replica restarted during a request (afterwards nothing was resident and it
had moved from us-west-2a to us-west-2c). Evidence, workspace cluster:
- 3 models resident plus the largest compound fresh, with Boltz-2 answering from its cache: no restart (replica
  peak 15.0 GB).
- One live Boltz-2 prediction (x7096a, 123 s) took its 32.4 GB L4 node to 18.5 GB used.
- On the Service the restart came while that compound's Boltz-2 affinity ran live; with 4 half-GPU slots on 2
  nodes, OF3Pool always shared a node with a Boltz2Live replica. 15 + 18.5 GB is past Ray's memory monitor
  threshold (95% of the node). The Service's logs need an API key this workspace lacks, so the kill itself is
  inferred.

Fix (75d1394): OF3Pool asks for a whole L4 (`num_gpus: 1`), so Boltz-2 (half an L4) runs on the other node;
Boltz2Live runs 1 replica. Verified on the Service by the both-live step above.

## Zones

Final run: OF3Pool and Boltz2Live both in us-west-2c (the Service's 2 L4 nodes landed in one zone this time).
Earlier Service versions tonight had OF3Pool in us-west-2a and in us-west-2c. On the workspace cluster one run
put OF3Pool in us-west-2a and Boltz2Live in us-west-2b: one app, GPUs from 2 zones. The Service may take GPUs
from any zone in the region (`enable_cross_zone_scaling: true`); it does not spread them on purpose.

## Load test (load_room.py 40×6)

240 requests per run over the 6 cached demo compounds, model ids base and the route. Every answer came from the
cache in every run, so the before tail is queueing, not compute.

| | P50 (s) | P95 (s) | P99 (s) | errors |
|---|---|---|---|---|
| before: version ibhuigyf (4f6af6e: Viewer 1 replica, max_ongoing_requests 5; OF3Pool loads a model before checking the cache; Boltz2Live sync) | 10.39 | 114.57 | 118.25 | 0 |
| after 1: ya7hppeb (ddb0700: Viewer 1 to 3 replicas, max_ongoing_requests 100, scoring in a thread; cached poses answered with no model or lock; MAX_RESIDENT 3) | 10.64 | 12.23 | 12.42 | 0 |
| after 2, first burst: g9yshm5m (d06d552: scoring memoized per pose file; Viewer starts at 2 replicas) | 2.05 | 4.36 | 5.14 | 0 |
| after 2, second burst (scores memoized on both replicas) | 0.16 | 0.20 | 0.24 | 0 |
| final, demo_1 applied (bsqwk14c), first burst | 0.15 | 1.73 | 2.39 | 0 |
| final, second burst | 0.15 | 0.30 | 0.32 | 0 |

After 1 removed the tail. Its P50 of 10.6 s was the Viewer's CPU: 240 requests in 65 s, about 0.25 s of RDKit
scoring per request on 1 replica, repeated for the same 12 poses. The burst of 240 took 166 s before, 65 s after
1, 15 s after 2 and about 1 s warm. A new molecule typed into the viewer still runs both models live (1 to 2
minutes, one at a time per model).

## Metrics: click path from the Anyscale Service page to the Serve panels

Not verified: this workspace has no browser. Expected: the Service's page in the Anyscale console → its metrics
(Grafana) → the **Serve** dashboard for "QPS per deployment", "P50/P90/P99 latency per deployment", "Queue size
per deployment" and "Replicas per deployment", and the **Serve Deployment** dashboard for "Multiplexed models per
replica", "Multiplexed model loads per replica" and "Multiplexed model cache hit rate" (exact panel names from
Ray 2.59). Please click through once and correct the path in 03 if it differs.

## Inconsistencies to tell Alicia (NB01/NB02)

- **The 145k base model (`of3p2`) has no stored results in this workspace**: `/mnt/shared_storage/aiddd/results_base-of3p2/`
  is empty, so `wh.load_results` has no `of3p2_*` columns. NB01's "OpenFold3 145k 84%" on SARS-CoV-2 follow-ons and
  "36%" on EV-A71 2A, and NB02's "145k 88% / 24%" table, cannot be checked against stored tables here.
- **x7259a, base model: NB01 §3 says 3.7 Å; every run tonight gives about 5 Å.** Same predictor class
  (`OF3Predictor`, legacy OpenFold3 0.4.3, 1 diffusion sample, seed 42), scratch outputs:

  | Run | RMSD (Å) |
  |---|---|
  | x7259a alone, run 1 | 5.25 |
  | x7259a alone, run 2 | 5.05 |
  | x7259a in NB01 §3's batch of 2 (with x7025a) | 5.25 |
  | Live pool / Service (in-process model), 3 runs | 5.20, 5.20, 5.33 |
  | 5 diffusion samples, top-ranked | 2.07 |

  Batch composition does not explain it; run to run varies by about 0.2 Å. NB01's 3.7 Å is unexplained; it may
  come from an earlier code state (for example a run before `OF3Predictor` named the MSA file so OpenFold3 reads
  it).
- **x7101a, ev71-v1 (`ft-p2-evfrag-1000s-8g-1023-0139`): Alicia's Slack table says 1.18 Å; with 1 sample tonight
  4.3 to 5.5 Å** (alone 5.04; in a batch of 8 late follow-ons 5.47; `run_openfold_pose` 5.05; Service path 4.32).
  **Cause: diffusion samples.** With 5 samples, top-ranked: ev71-v1 **1.19 Å** (Alicia 1.18), ev71-v2 **0.82 Å**
  (Alicia 0.80). Her base 6.42 matches 1 sample (6.17), not 5 (1.45), so her table mixes settings. Worth asking:
  which run (`-0139` or `-0140`, the example in `workshop_helpers.OURS_RUN`'s comment) and which scoring script.
