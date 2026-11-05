#!/bin/bash
# Sut's grasped start states (1000 train, 500 held-out eval) are no longer tracked at the tip of
# data_setup, but they are in its history. Run from the repo root.
set -euo pipefail

COMMIT=7ec44ae  # "1000 starts states and 500 eval states for Rl and residual learning"
git fetch origin data_setup
for split in train eval; do
    mkdir -p "start_states/$split"
    git show "$COMMIT:start_states/$split/states.pt" > "start_states/$split/states.pt"
    git show "$COMMIT:start_states/$split/generation_config.json" > "start_states/$split/generation_config.json"
done
ls -l start_states/train/states.pt start_states/eval/states.pt
