#!/usr/bin/env bash
# End-to-end run on the bundled KuaiRec_example subset: data processing -> training -> evaluation.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
python3 dataprocessing.py --stage all \
    --data-dir data/KuaiRec_example --out-dir processed/kuairec_example \
    --kcore-k 2 --l-min-days 3 --burn-in-days 1
python3 -m perl.train    --config configs/kuairec_example.yaml
python3 -m perl.evaluate --config configs/kuairec_example.yaml
