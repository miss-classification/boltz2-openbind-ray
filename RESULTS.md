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
- MAX_RESIDENT on the Service: 3 (Geoff chose 3 on Oct 25 at 21:40, before going to sleep).

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

## Zones: Boltz2Live <zone>, OF3Pool <zone>

## Load test (load_room.py 40×6)
| | P50 | P95 | P99 | errors |
|---|---|---|---|---|
| before | | | | |
| after | | | | |

## Metrics: click path from the Anyscale Service page to the Serve panels

## Inconsistencies to tell Alicia (NB01/NB02)
