#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 ENTITY/PROJECT/SWEEP_ID [COUNT]"
    exit 1
fi

SWEEP_ID="$1"
COUNT="${2:-30}"

if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: COUNT must be a positive integer."
    exit 1
fi

echo "Starting W&B agent"
echo "Sweep: $SWEEP_ID"
echo "Maximum runs: $COUNT"

wandb agent --count "$COUNT" --forward-signals "$SWEEP_ID"
