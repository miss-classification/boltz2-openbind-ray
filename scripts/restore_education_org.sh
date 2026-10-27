#!/usr/bin/env bash
# Restore the workshop data in a new Anyscale org (the education org) from the
# aiddd.tar.gz archive of /mnt/shared_storage/aiddd, then rebuild the image.
#
#     bash scripts/restore_education_org.sh '<download link>'            # data + image
#     bash scripts/restore_education_org.sh '<download link>' --deploy   # also deploy aiddd-viewer
#     bash scripts/restore_education_org.sh '<download link>' --skip-image
#
# Run it from a workspace terminal in the new org, inside a clone of this repo, together with
# `python scripts/setup_data.py` (before or after). The archive holds only what setup_data.py
# does not download: Boltz-2, the public OpenFold3 checkpoints and the EV-A71 release come from
# setup_data.py. It never overwrites a file that already exists.
# Safe to re-run: the download resumes, and a finished step is skipped.
# If the link has expired (HTTP 403), ask for a fresh one and run the same command again.
set -euo pipefail

EXPECTED_SHA256="0672088b4382740b969b126bec89773605abe66daea4b63404daff950029ea65"
EXPECTED_ENTRIES="19157"
IMAGE_NAME="aiddd-boltz2-of3"

URL="${1:-}"
[[ -z "$URL" || "$URL" == --* ]] && { echo "usage: $0 '<download link>' [--deploy] [--skip-image]"; exit 2; }
shift
DEPLOY=0; SKIP_IMAGE=0
for a in "$@"; do
  case "$a" in
    --deploy) DEPLOY=1 ;;
    --skip-image) SKIP_IMAGE=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL=/mnt/local_storage/aiddd.tar.gz
DEST=/mnt/shared_storage
DONE="$DEST/aiddd/.restored"

[[ -d "$DEST" ]] || { echo "No $DEST in this workspace: the cloud needs shared storage."; exit 1; }

if [[ -f "$DONE" ]]; then
  echo "1-3. Data already restored in $DEST/aiddd"
else
  echo "1. Downloading to $TARBALL (resumes if interrupted)"
  if ! curl -fL -C - --retry 5 --retry-delay 5 -o "$TARBALL" "$URL"; then
    echo "Download failed. If it says 403, the link expired: get a fresh one and re-run this command."
    exit 1
  fi

  echo "2. Checking the archive is intact"
  echo "$EXPECTED_SHA256  $TARBALL" | sha256sum -c -

  # Works on an empty folder or over a `python scripts/setup_data.py` install: the files that
  # script downloads are identical to the ones in the archive, so those are kept, not rewritten.
  echo "3. Unpacking to $DEST/aiddd (adds what setup_data.py does not download)"
  if command -v pigz >/dev/null; then Z=(-I pigz); else Z=(-z); fi
  tar "${Z[@]}" -xf "$TARBALL" -C "$DEST" --skip-old-files
  n=$(find "$DEST/aiddd" | wc -l)
  (( n >= EXPECTED_ENTRIES )) || { echo "Expected at least $EXPECTED_ENTRIES files and folders, found $n"; exit 1; }
  for f in of3_cache/of3p2-ft-ev2a-weights.pt \
           finetune/runs/ft-p2-evfrag-1000s-8g-1023-0139/checkpoint_2026-10-23_04-05-55.886640/of3_ev71_ft.pt \
           finetune/runs/tune-of3-ev71/summary.csv tune/runs/tune-boltz2-ev71/summary.csv \
           predictions/live_of3ft_ev71-v2 results_ft-p2-evfrag80-1000s-8g-32samp-1023-0203 \
           raw/SARS-CoV-2_Mpro/structures workshop/live; do
    [[ -e "$DEST/aiddd/$f" ]] || { echo "Missing after unpack: $f"; exit 1; }
  done
  touch "$DONE"
  echo "   $n files and folders"
  rm -f "$TARBALL"
fi

if [[ $SKIP_IMAGE == 0 ]]; then
  echo "4. Building the $IMAGE_NAME image in this org (about 15-30 min)"
  anyscale image build -f "$REPO/image/Containerfile" -n "$IMAGE_NAME"
  URI=$(anyscale image get -n "$IMAGE_NAME" -j | python -c 'import json,sys; print(json.load(sys.stdin)["latest_image_uri"])')
  sed -i -E "s#anyscale/image/$IMAGE_NAME:[0-9]+#$URI#" "$REPO"/jobs/*.yaml
  echo "   jobs/*.yaml now use $URI"
fi

if [[ $DEPLOY == 1 ]]; then
  echo "5. Deploying the aiddd-viewer Service"
  (cd "$REPO" && anyscale service deploy -f jobs/service.yaml && anyscale service wait -n aiddd-viewer --state RUNNING)
fi

for f in boltz_cache/boltz2_conf.ckpt of3_cache/of3-p2-145k.pt raw/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv; do
  [[ -e "$DEST/aiddd/$f" ]] || { echo; echo "Not there yet: $f. Run: python scripts/setup_data.py"; break; }
done

cat <<EOF

Done. Next:
  - Switch this workspace to the ${URI:-$IMAGE_NAME} image (Workspace settings), then open the notebooks.
  - Optional: pin the fine-tune notebook 02 shows as ours:
      export AIDDD_OURS_RUN=ft-p2-evfrag80-1000s-8g-32samp-1023-0203
  - README.md and scripts/load_room.py, scripts/measure_service.py still name the old Service URL;
    notebook 03 finds the Service by name, so it does not need it.
EOF
