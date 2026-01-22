#!/usr/bin/env bash
set -euo pipefail

# One-command end-to-end run.
# Example:
#   bash scripts/run_full_pipeline.sh --langs "bn,hi,ha,ma,ur,uz,kk" --gpu 0

LANGS=""
GPU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --langs)
      LANGS="$2"; shift 2;;
    --gpu)
      GPU="$2"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [[ -z "${LANGS}" ]]; then
  echo "Error: --langs is required, e.g. --langs "bn,hi,ha,ma"" >&2
  exit 1
fi

# Paths (override via env vars if desired)
IMAGES_DIR="${SILKROAD_IMAGES_DIR:-data/images}"
VAL_DIR="${SILKROAD_IMAGENET_VAL_DIR:-data/images/ILSVRC2012_img_val}"

OUT_FEATURES="${SILKROAD_SIGLIP_FEATS:-outputs/features/siglip_features.pt}"
OUT_MANIFEST="${SILKROAD_MANIFEST:-outputs/manifest/manifest.json}"
OUT_CAPTIONS="${SILKROAD_CAPTIONS:-outputs/captions/generated_ranked.jsonl}"
OUT_FILTERED="${SILKROAD_FILTERED_CAPTIONS:-outputs/captions/filtered.json}"
OUT_TRANSLATIONS="${SILKROAD_TRANSLATIONS:-outputs/translations/translations.json}"
OUT_BT="${SILKROAD_BACKTRANSLATIONS:-outputs/translations/translations_bt.json}"
OUT_SCORED="${SILKROAD_SCORED:-outputs/scored/scored.json}"
OUT_SCORED_UZ="${SILKROAD_SCORED_UZ:-outputs/scored/scored_no_uz_cyrillic.json}"
OUT_GOLDEN="${SILKROAD_GOLDEN:-outputs/final/golden.json}"
OUT_SPLITS="${SILKROAD_SPLITS_DIR:-outputs/splits}"

mkdir -p outputs/features outputs/manifest outputs/captions outputs/translations outputs/scored outputs/final "${OUT_SPLITS}" logs

echo "[Stage 00] Extract image features (SigLIP)"
python pipelines/00_extract_image_features.py   --image_root "${VAL_DIR}"   --save_path "${OUT_FEATURES}"

echo "[Stage 01] Build manifest"
python pipelines/01_build_manifest_json.py   --dataset_root "${VAL_DIR}"   --output_path "${OUT_MANIFEST}"

echo "[Stage 02] Generate & rank captions"
python pipelines/02_generate_and_rank.py   --manifest "${OUT_MANIFEST}"   --output_file "${OUT_CAPTIONS}"   --feature_file "${OUT_FEATURES}"   --gpu_id "${GPU}"

echo "[Stage 04] Filter captions"
python pipelines/04_filter_dataset.py   --input_file "${OUT_CAPTIONS}"   --output_file "${OUT_FILTERED}"

echo "[Stage 06] Translate (4 models)"
python pipelines/06_translate_all_lang_4model.py   --input_file "${OUT_FILTERED}"   --output_file "${OUT_TRANSLATIONS}"   --langs "${LANGS}"   --gpu_id "${GPU}"

echo "[Stage 07] Back-translate"
python pipelines/07_b_back_translate.py   --input_file "${OUT_TRANSLATIONS}"   --output_file "${OUT_BT}"   --gpu_id "${GPU}"

echo "[Stage 08] Quality estimation / scoring"
python pipelines/08_quality_estimation.py   --input_file "${OUT_BT}"   --output_file "${OUT_SCORED}"   --gpu_id "${GPU}"

echo "[Stage 09] Filter Cyrillic Uzbek (optional but used in paper)"
python pipelines/09_filter_uzbek_cyrillic.py   --input_file "${OUT_SCORED}"   --output_file "${OUT_SCORED_UZ}"

echo "[Stage 10] Select best translations + final filtering"
python pipelines/10_QE_filter.py   --input_file "${OUT_SCORED_UZ}"   --output_file "${OUT_GOLDEN}"

echo "[Stage 11] Split by language"
python pipelines/11_split_languages.py   --input_file "${OUT_GOLDEN}"   --output_dir "${OUT_SPLITS}"

echo "✅ Done. Outputs are under: outputs/"
