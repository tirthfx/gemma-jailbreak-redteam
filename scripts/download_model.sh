#!/bin/bash
# Robust model download for a slow/flaky network.
#
# huggingface_hub's plain-HTTP downloader writes to a `<hash>.incomplete` file and
# resumes via range requests when re-invoked — so on failure, the fix is just to
# retry the same call, not to restart from zero. This wraps that in a bounded
# retry loop so a single dropped connection doesn't require babysitting.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_DISABLE_XET=1  # see src/model.py for why

MAX_ATTEMPTS=30
SLEEP_BETWEEN=15

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[download] attempt $attempt/$MAX_ATTEMPTS ($(date '+%H:%M:%S'))"
    .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-3-4b-it')
print('DOWNLOAD_COMPLETE')
"
    if [ $? -eq 0 ]; then
        echo "[download] complete."
        exit 0
    fi
    echo "[download] attempt $attempt failed, retrying in ${SLEEP_BETWEEN}s..."
    sleep "$SLEEP_BETWEEN"
done

echo "[download] gave up after $MAX_ATTEMPTS attempts."
exit 1
