#!/usr/bin/env bash
set -euo pipefail
PY=/home/mcy/.conda/envs/fin_research/bin/python
ENTRY=/home/mcy/Projects/GLANCE/repro/scienceqa_image/full_epoch/run.py
if [[ "${1:-}" == "--background" ]]; then
    shift
    # Create an idle shell first, so retain-on-exit is set before Python starts.
    pane_id=$(tmux new-session -d -P -F '#{pane_id}' -s glance-sqa-full)
    tmux set-option -w -t "$pane_id" remain-on-exit on
    exec tmux respawn-pane -k -t "$pane_id" "$PY" "$ENTRY" --execute "$@"
fi
exec "$PY" "$ENTRY" "$@"
