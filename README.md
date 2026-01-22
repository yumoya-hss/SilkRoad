# Anonymous Reproducibility Package — Dataset Construction Code

This repository contains the full pipeline used in our paper to **construct and filter a multilingual multimodal dataset**
from images and automatically generated captions.

**Design goals**
- **End-to-end reproducibility**: deterministic stages, clear inputs/outputs, and fixed thresholds.
- **Anonymity-friendly**: no hard-coded personal paths, no author identifiers, no git history required.
- **Top-conference style release**: standard layout, documented CLI, and reproducibility checklist.

---

## Repository layout

```
pipelines/              # Main end-to-end dataset construction pipeline (stages 00–11)
experiments/            # Ablations/analysis scripts used in the paper (optional)
configs/                # Default configs & paper thresholds
scripts/                # One-command runners
data/
  images/               # Place images here (not tracked)
  metadata/             # Place metadata/label files here (not tracked)
models/                 # Optional local model checkpoints (not tracked)
outputs/                # All pipeline outputs (not tracked)
logs/                   # Logs (not tracked)
```

---

## Quickstart

### 0) Environment
We recommend Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> If you run in an offline cluster: pre-download required checkpoints and set `HF_HOME` / `TRANSFORMERS_OFFLINE=1`.

### 1) Prepare inputs

#### Images
Put images under:
- `data/images/` (single directory), **or**
- `data/images/ILSVRC2012_img_val/` if you use ImageNet val naming.

#### Metadata (if using ImageNet-50K)
Place the following under `data/metadata/`:
- `imagenet_2012_validation_synset_labels.txt`
- `imagenet_class_index.json`

### 2) Run the full pipeline

```bash
bash scripts/run_full_pipeline.sh \
  --langs "bn,hi,ha,ma,ur,uz,kk" \
  --gpu 0
```

All artifacts will be written under `outputs/` (see "Outputs" below).

---

## Outputs

Typical outputs (paths may vary if you override env vars):

- `outputs/features/siglip_features.pt` — image features
- `outputs/manifest/manifest.json` — image manifest with class/name info
- `outputs/captions/generated_ranked.jsonl` — generated caption candidates + ranking
- `outputs/captions/filtered.json` — filtered captions by thresholds
- `outputs/translations/translations.json` — multilingual translations (4 models)
- `outputs/scored/scored.json` — QE + text similarity + image-text scores
- `outputs/final/golden.json` — final filtered dataset with selected best translations
- `outputs/splits/<lang>/{short,long}.json` — per-language split files for release

---

## Reproducing paper thresholds

Paper thresholds are recorded in:
- `configs/paper_thresholds.yaml`

The filtering scripts implement these thresholds directly; you can adjust them via CLI/env vars as described in the scripts.

---

## Notes on models

By default, scripts use Hugging Face model IDs via `from_pretrained(...)`.
You can override any model by setting environment variables, e.g.:

```bash
export SILKROAD_SIGLIP_MODEL="google/siglip-so400m-patch14-384"
export SILKROAD_VL_CAPTION_MODEL="Qwen/Qwen3-VL-8B-Instruct"
export SILKROAD_NLLB_MODEL="facebook/nllb-200-3.3B"
export SILKROAD_SEAMLESS_MODEL="facebook/seamless-m4t-v2-large"
export SILKROAD_MADLAD_MODEL="google/madlad400-7b-mt"
export SILKROAD_QWEN_TRANSLATOR="Qwen/Qwen3-32B-Instruct"
```

If you use **local checkpoints**, point the env var to your local directory under `models/`.

---

## Citation

See `CITATION.cff`. If your venue requires BibTeX, add it to the paper appendix and keep this file consistent.

---

## License

This code is released under the **Apache-2.0 License** (see `LICENSE`).

---

## Support for anonymous review

If you plan to host on **4open**:
- Do **not** push `data/`, `models/`, `outputs/`, `logs/` to GitHub.
- Make sure your git commits use a non-identifying author/email.
- Add any remaining sensitive tokens to 4open's anonymization list.

See `docs/ANONYMIZATION.md`.
