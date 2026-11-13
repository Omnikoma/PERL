# PERL

PERL (Progressive Evidence Refinement over the Item Lifecycle) is a model-agnostic framework for item cold-start recommendation. It initialises a new item's representation from a lifecycle-aligned multi-prototype population prior and progressively refines it with the item's own behavioural trajectory (encoded by a LoRA-fine-tuned TimesFM 2.5 time-series foundation model) and its accumulating collaborative feedback, with learned reliability scores gating each refinement step. This repository is the official implementation of *"Bridging Cold-to-Warm Recommendation: Progressive Evidence Refinement over Item Lifecycles"*.

## Quick Start

- Install the dependencies by `pip install -r requirements.txt`.
- Download the TimesFM 2.5 weights by `huggingface-cli download google/timesfm-2.5-200m-pytorch --local-dir weights/timesfm-2.5-200m-pytorch`.
- Run `bash run_example.sh` to process, train and evaluate on the bundled example subset in `data/KuaiRec_example/`.

## Run From Scratch

1. Pre-process the dataset
   - Download [KuaiRec](https://kuairec.com/) and put `big_matrix.csv`, `item_daily_features.csv`, `item_categories.csv` and `kuairec_caption_category.csv` in `data/KuaiRec/`.
   - Build the model-ready data by `python3 dataprocessing.py --stage all`. The processed results will be stored in `processed/kuairec/`.

2. Prepare the pretrained temporal encoder
   - Download the TimesFM 2.5 weights by `huggingface-cli download google/timesfm-2.5-200m-pytorch --local-dir weights/timesfm-2.5-200m-pytorch`. The weights will be stored in `weights/timesfm-2.5-200m-pytorch/` and loaded offline.

3. Train PERL
   - Go back to the root directory.
   - Train PERL by running `PYTHONPATH=. python3 -m perl.train --config configs/kuairec.yaml`. The best checkpoint will be stored in `checkpoints/kuairec/perl_best.pt`.

4. Evaluate PERL
   - Evaluate PERL by running `PYTHONPATH=. python3 -m perl.evaluate --config configs/kuairec.yaml`. The results across the four lifecycle stages will be printed and stored in `processed/kuairec/eval_report.json`.

## Citation

```bibtex
@article{xu2026perl,
  title   = {Bridging Cold-to-Warm Recommendation: Progressive Evidence Refinement over Item Lifecycles},
  author  = {Xu, Beibei and Guo, XiaoBo and Wang, Binfeng and Liu, Xiyu},
  journal = { },
  year    = {2026}
}
```

## Acknowledgements

The TimesFM 2.5 package under `third_party/timesfm/` is from [google-research/timesfm](https://github.com/google-research/timesfm) (Apache-2.0). KuaiRec is released by its authors under their own terms.
