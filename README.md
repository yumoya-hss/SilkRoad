SilkRoad-MMT: Bridging the Low-Resource Gap in Multimodal Machine Translation

📖 Introduction

SilkRoad-MMT is a comprehensive pipeline and dataset designed to address the critical data scarcity in Multimodal Machine Translation (MMT) for low-resource languages in Central and South Asia. This project focuses on six under-represented languages: Uyghur, Kazakh, Uzbek, Kyrgyz, Tajik, and Urdu.

We propose a fully automated, high-quality dataset construction framework that integrates:

Metric-Driven Visual Captioning: Utilizing Qwen3-VL to generate high-fidelity English captions, filtered by SigLIP for visual grounding.

Hybrid Ensemble Translation: A novel ensemble strategy combining NLLB-200, SeamlessM4T, MADLAD-400, and Qwen-LLM to balance lexical precision and linguistic fluency.

Tri-QE Filtering Protocol: A rigorous triangular quality estimation mechanism evaluating candidates via COMET-Kiwi (Quality), BERTScore (Consistency), and CLIP (Visual Dependency).

This repository contains all the scripts required to reproduce the dataset construction pipeline, from image feature extraction to final quality scoring.

🛠️ Prerequisites

The pipeline is optimized for high-performance computing environments (e.g., NVIDIA H100 clusters) and supports fully offline execution.

Core Dependencies

pip install torch torchvision torchaudio
pip install transformers peft bitsandbytes accelerate
pip install unbabel-comet bert-score sacrebleu
pip install pillow tqdm pandas openpyxl


Environment Variables (For Offline Mode)

If you are running this in an air-gapped environment, please set the following environment variables to prevent Hugging Face libraries from attempting to connect to the internet:

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false


📂 Directory Structure

SilkRoad-MMT-Pipeline/
├── scripts/
│   ├── 00_extract_image_features.py   # Pre-extract image features using SigLIP
│   ├── 01_build_manifest_json.py      # Build initial image metadata index
│   ├── 02_generate_and_rank.py        # Generate captions via Qwen3-VL & Rank via SigLIP
│   ├── 03_analyze_caption.py          # Statistical analysis of generated captions
│   ├── 04_filter_dataset.py           # Initial filtering based on thresholds
│   ├── 05_json_convert.py             # JSON format validation and repair
│   ├── 06_translate_all_lang_4model.py # Core: Hybrid Ensemble Translation
│   ├── 07_b_back_translate.py         # Back-translation (Target -> English)
│   ├── 08_quality_estimation.py       # Tri-QE Scoring (COMET + BERT + CLIP)
│   ├── 09_filter_uzbek_cyrillic.py    # Language-specific cleaning (Script correction)
│   └── 11_split_languages.py          # Split final dataset by language and length
├── data/              # Intermediate JSON files (Not included in repo)
├── models/            # Local model checkpoints (Not included in repo)
└── README.md


🚀 Usage Pipeline

Follow these steps sequentially to construct the SilkRoad-MMT dataset.

Phase 1: Visual Captioning

Feature Extraction: Pre-compute image features to accelerate subsequent ranking.

python scripts/00_extract_image_features.py --image_root /path/to/images


Generation & Ranking: Generate candidate captions (Short/Long) using Qwen3-VL and select the best ones using SigLIP.

python scripts/02_generate_and_rank.py


Initial Filtering: Filter out low-quality captions based on SigLIP scores and length constraints.

python scripts/04_filter_dataset.py --input_file raw_captions.json --output_file filtered_captions.json


Phase 2: Ensemble Translation

Hybrid Translation: Translate the English captions into 6 target languages using the 4-model ensemble.

python scripts/06_translate_all_lang_4model.py --input_file filtered_captions.json --output_file dataset_translated.json


Phase 3: Tri-QE Filtering

Back-Translation: Translate the target language text back to English using NLLB-200.

python scripts/07_b_back_translate.py --input_file dataset_translated.json


Quality Estimation (Tri-QE): Calculate COMET, BERTScore, and CLIP scores.
Note: This script includes Monkey Patches for offline compatibility with comet and transformers.

python scripts/08_quality_estimation.py --input_file dataset_with_bt.json


Post-Processing: Clean up script errors (e.g., removing Cyrillic characters from Uzbek translations) and split the dataset.

python scripts/09_filter_uzbek_cyrillic.py
python scripts/11_split_languages.py


⚠️ Important Notes

Path Configuration: The scripts currently use hardcoded paths (e.g., /mnt/raid/...). You must update these paths in the "Configuration" section at the top of each script to match your local environment before running.

Model Weights: Ensure you have downloaded the following models to your local directory:

Qwen/Qwen2.5-VL-7B-Instruct (or Qwen3-VL)

google/siglip-so400m-patch14-384

facebook/nllb-200-3.3B

facebook/seamless-m4t-v2-large

google/madlad400-7b-mt

Unbabel/wmt22-cometkiwi-da

openai/clip-vit-large-patch14

Hardware: We recommend using GPUs with at least 24GB VRAM for inference. For Qwen-VL generation, A100/H100 GPUs are preferred.

📜 Citation

If you use this code or dataset in your research, please cite our paper:

@inproceedings{silkroad2025,
  title={SilkRoad-MMT: Bridging the Low-Resource Gap in Multimodal Machine Translation},
  author={Anonymous},
  booktitle={ACL/EMNLP},
  year={2025}
}


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
