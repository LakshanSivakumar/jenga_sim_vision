#!/bin/zsh
# Generate the v2 dataset, then run the EDA notebook on it, then the frozen test set.
#
#   nohup caffeinate -ims zsh tools/run_v2.sh &
#
# Safe to re-run after an interruption: every generate step uses --resume.
# Progress: tail -f data/v2_run.log
set -e
cd "$(dirname "$0")/.."
LOG=data/v2_run.log
UV=${UV:-$HOME/.local/bin/uv}
stamp() { echo "\n=== $(date '+%Y-%m-%d %H:%M') $* ===" >> $LOG; }

stamp "training set: 400 towers"
.venv/bin/python generate.py --towers 400 --views 6 --seed 0 --out data/v2_train --resume >> $LOG 2>&1

stamp "EDA notebook"
(cd notebooks && JENGA_DATA=../data/v2_train JENGA_FIGS=../docs/eda_v2 \
  $UV run --no-project --with nbconvert --with ipykernel --with pandas --with numpy \
      --with matplotlib --with pillow \
  jupyter nbconvert --to notebook --execute DAP_IEDA_Jenga.ipynb --output DAP_IEDA_Jenga.ipynb \
      --ExecutePreprocessor.timeout=3600) >> $LOG 2>&1 || stamp "EDA notebook FAILED (see above)"

stamp "proposal-style charts"
.venv/bin/python tools/eda_figures.py data/v2_train --out docs/eda_v2_proposal >> $LOG 2>&1 \
  || stamp "proposal charts FAILED (see above)"

stamp "test set: 100 towers, seed 99 (frozen: do not explore)"
.venv/bin/python generate.py --towers 100 --views 6 --seed 99 --out data/v2_test --resume >> $LOG 2>&1

stamp "ALL DONE"
