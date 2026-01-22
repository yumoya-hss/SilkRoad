import os
import json
import shutil
import torch
import math
import sacrebleu
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml
import logging

# ================= 🔴 1. 强制离线环境设置 🔴 =================
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ================= 🔴 2. 路径配置 🔴 =================
BASELINE_FILE = "pred_baseline.json"
FINETUNED_FILE = "pred_finetuned_v2_3epoch.json"
OUTPUT_CSV = "final_experiment_results_full.csv"

# 1. CometKiwi
KIWI_ROOT = "/mnt/raid/hss/model/wmt22-cometkiwi-da"
INFOXLM_PATH = "/mnt/raid/hss/model/infoxlm-large"

# 2. BERTScore (XLM-R)
XLMR_PATH = "/mnt/raid/hss/model/xlm-roberta-large"

# 3. ✅ PPL (改为 Multilingual BERT)
# 假设您的 mBERT 路径如下，如果没有，可改为 XLMR_PATH (XLM-R 也是 BERT 架构)
PPL_MODEL_PATH = "/mnt/raid/hss/model/xlm-roberta-large"

# 4. Visual Models
SIGLIP_PATH = "/mnt/raid/hss/model/siglip-so400m-patch14-384"
CLIP_BASE_PATH = "/mnt/raid/hss/model/clip-vit-base-patch32"
CLIP_LARGE_PATH = "/mnt/raid/hss/model/clip-vit-large-patch14"
# =========================================================

# -----------------------------------------------------------------------------
# 🛠️ 步骤 0: 物理修复
# -----------------------------------------------------------------------------
def check_and_fix_files():
    print("🔧 [Step 0] Checking files...")
    
    # 1. InfoXLM
    if not os.path.exists(INFOXLM_PATH): os.makedirs(INFOXLM_PATH, exist_ok=True)
    sp_path = os.path.join(INFOXLM_PATH, "sentencepiece.bpe.model")
    if not os.path.exists(sp_path):
        src = os.path.join(XLMR_PATH, "sentencepiece.bpe.model")
        if os.path.exists(src): shutil.copy(src, sp_path)
    
    tok_conf = os.path.join(INFOXLM_PATH, "tokenizer_config.json")
    with open(tok_conf, 'w') as f:
        json.dump({"do_lower_case": False, "unk_token": "<unk>", "sep_token": "</s>", 
                   "pad_token": "<pad>", "cls_token": "<s>", "mask_token": "<mask>", 
                   "model_type": "xlm-roberta", "use_fast": False}, f)

    # 2. Kiwi hparams
    yaml_path = os.path.join(KIWI_ROOT, "hparams.yaml")
    if not os.path.exists(yaml_path): yaml_path = os.path.join(KIWI_ROOT, "checkpoints", "hparams.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f: config = yaml.safe_load(f) or {}
        if config.get("encoder_model") != INFOXLM_PATH:
            config["encoder_model"] = INFOXLM_PATH
            config["pretrained_model"] = INFOXLM_PATH
            config["load_weights_from_checkpoint"] = True
            with open(yaml_path, 'w') as f: yaml.dump(config, f)

    # 3. CLIP Config
    for clip_path in [CLIP_BASE_PATH, CLIP_LARGE_PATH]:
        if os.path.exists(clip_path):
            prep_conf = os.path.join(clip_path, "preprocessor_config.json")
            if not os.path.exists(prep_conf):
                dummy_prep = {
                    "crop_size": 224, "do_center_crop": True, "do_convert_rgb": True, "do_normalize": True,
                    "do_resize": True, "feature_extractor_type": "CLIPFeatureExtractor",
                    "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
                    "resample": 3, "size": 224
                }
                try: 
                    with open(prep_conf, 'w') as f: json.dump(dummy_prep, f)
                except: 
                    pass

# -----------------------------------------------------------------------------
# 🧙‍♂️ 步骤 1: 智能路由拦截器
# -----------------------------------------------------------------------------
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, 
    AutoModelForCausalLM, AutoModelForMaskedLM, # ✅ 引入 MaskedLM
    XLMRobertaTokenizer, XLMRobertaTokenizerFast, XLMRobertaModel, XLMRobertaConfig,
    CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
)

GLOBAL_TOKENIZERS = {}

def install_smart_router():
    print("🧙‍♂️ [Step 1] Installing Smart Router Interceptor...")
    try:
        GLOBAL_TOKENIZERS['infoxlm'] = XLMRobertaTokenizer(vocab_file=os.path.join(INFOXLM_PATH, "sentencepiece.bpe.model"))
    except: pass
    
    def router_interceptor(original_func, cls_name):
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            path_str = str(pretrained_model_name_or_path)
            
            # 路由逻辑
            if "infoxlm" in path_str.lower() or path_str == INFOXLM_PATH:
                if "Tokenizer" in cls_name and 'infoxlm' in GLOBAL_TOKENIZERS: return GLOBAL_TOKENIZERS['infoxlm']
                if not os.path.exists(path_str):
                    pretrained_model_name_or_path = INFOXLM_PATH
                    kwargs['local_files_only'] = True
            elif "xlm-roberta" in path_str.lower() and "infoxlm" not in path_str.lower():
                if not os.path.exists(path_str) or "huggingface.co" in path_str:
                    pretrained_model_name_or_path = XLMR_PATH
                    kwargs['local_files_only'] = True
            
            # ✅ 新增: PPL (mBERT)
            elif "bert-" in path_str.lower() or "multilingual" in path_str.lower():
                if not os.path.exists(path_str):
                    pretrained_model_name_or_path = PPL_MODEL_PATH
                    kwargs['local_files_only'] = True

            elif "clip" in path_str.lower() and "siglip" not in path_str.lower():
                if "base" in path_str.lower() and not os.path.exists(path_str):
                    pretrained_model_name_or_path = CLIP_BASE_PATH
                    kwargs['local_files_only'] = True
                elif "large" in path_str.lower() and not os.path.exists(path_str):
                    pretrained_model_name_or_path = CLIP_LARGE_PATH
                    kwargs['local_files_only'] = True
            
            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)
        return wrapper

    target_classes = [AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, XLMRobertaTokenizer, XLMRobertaTokenizerFast, XLMRobertaConfig, XLMRobertaModel, CLIPModel, CLIPProcessor]
    for cls in target_classes:
        if hasattr(cls, 'from_pretrained'):
            cls.from_pretrained = classmethod(router_interceptor(cls.from_pretrained.__func__, cls.__name__))

    print("   ✅ Smart Router Active.")

check_and_fix_files()
install_smart_router()

# =============================================================================
# 🚀 步骤 2: 通用手动加载函数
# =============================================================================
def load_manual_model(model_path, model_class, device, dtype=None):
    print(f"   -> Manual Load: {os.path.basename(model_path)} ({model_class.__name__})")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        
        # 初始化模型
        if hasattr(model_class, 'from_config'):
            model = model_class.from_config(config)
        else:
            model = model_class(config)
            
        if dtype: model = model.to(dtype)
            
        # 寻找权重
        safe_file = os.path.join(model_path, "model.safetensors")
        safe_index = os.path.join(model_path, "model.safetensors.index.json")
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")

        if os.path.exists(safe_file) or os.path.exists(safe_index):
            print("      🛡️ Safetensors detected...")
            model = model_class.from_pretrained(model_path, config=config, trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
        elif os.path.exists(bin_file):
            print("      🛡️ Pickle (.bin) detected. Using raw torch.load...")
            state_dict = torch.load(bin_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(bin_index):
            print("      ⚠️ Sharded .bin detected. Attempting standard load...")
            model = model_class.from_pretrained(model_path, config=config, trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
        else:
            raise FileNotFoundError("No model weights found.")

        return model.to(device).eval()

    except Exception as e:
        print(f"      ❌ Manual Load Failed: {e}")
        try:
            return model_class.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device).eval()
        except:
            return None

# =============================================================================
# 🚀 步骤 3: 加载模型
# =============================================================================
from comet import load_from_checkpoint
try:
    from bert_score import score as run_bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

def load_models():
    print("\n⏳ [Step 2] Loading Evaluation Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. CometKiwi
    kiwi_model = None
    try:
        ckpt_path = os.path.join(KIWI_ROOT, "checkpoints", "model.ckpt")
        if not os.path.exists(ckpt_path): ckpt_path = os.path.join(KIWI_ROOT, "model.ckpt")
        print(f"   -> Loading CometKiwi from: {ckpt_path}")
        kiwi_model = load_from_checkpoint(ckpt_path, strict=False)
        print("      ✅ CometKiwi Loaded!")
    except Exception as e:
        print(f"      ❌ CometKiwi Failed: {e}")

    # 2. ✅ PPL (mBERT) - AutoModelForMaskedLM
    ppl_bundle = None
    try:
        if os.path.exists(PPL_MODEL_PATH):
            tok = AutoTokenizer.from_pretrained(PPL_MODEL_PATH, trust_remote_code=True, use_fast=False)
            # 使用 AutoModelForMaskedLM
            mod = load_manual_model(PPL_MODEL_PATH, AutoModelForMaskedLM, device)
            if mod:
                ppl_bundle = (mod, tok)
                print("      ✅ Multilingual BERT Loaded!")
        else:
            print(f"      ❌ mBERT path not found: {PPL_MODEL_PATH}")
    except Exception as e:
        print(f"      ❌ mBERT Failed: {e}")

    # 3. Visual Models
    visual_bundle = {}
    try:
        if os.path.exists(SIGLIP_PATH):
            visual_bundle['siglip'] = (
                SiglipModel.from_pretrained(SIGLIP_PATH, local_files_only=True).to(device).eval(),
                SiglipProcessor.from_pretrained(SIGLIP_PATH, local_files_only=True)
            )
            print("      ✅ SigLIP Loaded!")
    except Exception as e: print(f"      ❌ SigLIP Failed: {e}")

    try:
        if os.path.exists(CLIP_BASE_PATH):
            mod = load_manual_model(CLIP_BASE_PATH, CLIPModel, device)
            proc = CLIPProcessor.from_pretrained(CLIP_BASE_PATH, local_files_only=True)
            if mod:
                visual_bundle['clip_base'] = (mod, proc)
                print("      ✅ CLIP-Base Loaded!")
    except Exception as e: print(f"      ❌ CLIP-Base Failed: {e}")

    try:
        if os.path.exists(CLIP_LARGE_PATH):
            mod = load_manual_model(CLIP_LARGE_PATH, CLIPModel, device)
            proc = CLIPProcessor.from_pretrained(CLIP_LARGE_PATH, local_files_only=True)
            if mod:
                visual_bundle['clip_large'] = (mod, proc)
                print("      ✅ CLIP-Large Loaded!")
    except Exception as e: print(f"      ❌ CLIP-Large Failed: {e}")

    return kiwi_model, ppl_bundle, visual_bundle

# =============================================================================
# 📊 步骤 4: 计算逻辑 (🔥 重写 PPL 计算为 Pseudo-Perplexity)
# =============================================================================
def calculate_pseudo_ppl(texts, model, tokenizer):
    """
    计算 MLM (如 BERT) 的伪困惑度 (Pseudo-Perplexity)
    """
    if not texts: return 0.0
    device = model.device
    mask_id = tokenizer.mask_token_id
    if mask_id is None: return 0.0 # 无法计算

    nlls = []
    
    # 逐句计算（或者小 Batch 计算）
    for text in texts:
        if len(text.strip()) == 0: continue
        
        # 1. 编码
        inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"] # shape: [1, seq_len]
        seq_len = input_ids.shape[1]
        
        # 忽略太短的句子
        if seq_len < 3: continue 

        # 2. 构造 Batch Masking
        # 我们要计算 P(token_i | other_tokens)，所以要构建 seq_len 个样本，每个样本 mask 掉第 i 个词
        # 为了速度，我们跳过 CLS(0) 和 SEP(-1)
        
        # 复制 seq_len-2 份
        repeat_ids = input_ids.repeat(seq_len-2, 1) # [seq_len-2, seq_len]
        
        # 创建对角线 mask 矩阵
        # mask 的位置索引是 1 到 seq_len-2
        for i in range(seq_len-2):
            repeat_ids[i, i+1] = mask_id
            
        # 3. 推理 (Batch Inference)
        # 注意显存，如果句子很长，repeat_ids 会很大，需要分块
        # 这里做一个简单的显存保护：如果 batch > 64，分块处理
        batch_size = 64
        total_loss = 0.0
        
        for i in range(0, repeat_ids.shape[0], batch_size):
            chunk_ids = repeat_ids[i:i+batch_size]
            
            with torch.no_grad():
                outputs = model(chunk_ids)
                logits = outputs.logits # [batch, seq_len, vocab_size]
            
            # 4. 提取被 Mask 位置的 Logits
            # chunk_ids 中，第 k 个样本的 mask 位置是 i + k + 1
            # 对应的真实 token 也是 input_ids[0, i + k + 1]
            
            for k in range(chunk_ids.shape[0]):
                token_idx = i + k + 1
                target_token_id = input_ids[0, token_idx]
                token_logits = logits[k, token_idx, :]
                
                # CrossEntropy = -log_softmax
                log_probs = torch.log_softmax(token_logits, dim=-1)
                total_loss += -log_probs[target_token_id].item()

        # 5. 平均 Loss
        avg_loss = total_loss / (seq_len - 2)
        nlls.append(avg_loss)

    if not nlls: return 0.0
    # PPL = exp(mean(losses))
    return math.exp(sum(nlls) / len(nlls))

def calculate_visual_score(img_paths, texts, model, processor, device):
    score_sum, count = 0, 0
    for img_path, txt in zip(img_paths, texts):
        try:
            if not os.path.exists(img_path): continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[txt[:64]], images=image, return_tensors="pt", padding="max_length", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                img_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                txt_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                score = (img_emb @ txt_emb.T).item()
                score_sum += score
                count += 1
        except: pass
    return score_sum / count if count > 0 else 0

def compute_metrics(df, kiwi_model, ppl_bundle, visual_bundle):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refs = df['ref'].tolist()
    hyps = df['hyp'].tolist()
    srcs = df['src'].tolist()
    img_paths = df['image_path'].tolist()

    kiwi = 0.0
    if kiwi_model:
        try:
            data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
            out = kiwi_model.predict(data, batch_size=32, gpus=1, progress_bar=False)
            kiwi = out.system_score
        except: pass

    bert = 0.0
    if HAS_BERTSCORE:
        try:
            P, R, F1 = run_bert_score(hyps, refs, model_type=XLMR_PATH, num_layers=17, device=device, batch_size=32, verbose=False)
            bert = F1.mean().item()
        except: pass

    ppl = 0.0
    if ppl_bundle:
        try: 
            # 🔥 使用 Pseudo-PPL 计算
            ppl = calculate_pseudo_ppl(hyps, ppl_bundle[0], ppl_bundle[1])
        except: pass

    sig_score, clip_base, clip_large = 0.0, 0.0, 0.0
    if 'siglip' in visual_bundle: sig_score = calculate_visual_score(img_paths, hyps, visual_bundle['siglip'][0], visual_bundle['siglip'][1], device)
    if 'clip_base' in visual_bundle: clip_base = calculate_visual_score(img_paths, hyps, visual_bundle['clip_base'][0], visual_bundle['clip_base'][1], device)
    if 'clip_large' in visual_bundle: clip_large = calculate_visual_score(img_paths, hyps, visual_bundle['clip_large'][0], visual_bundle['clip_large'][1], device)

    return ppl, bert, kiwi, sig_score, clip_base, clip_large

def main():
    if not os.path.exists(BASELINE_FILE) or not os.path.exists(FINETUNED_FILE):
        print(f"❌ Input files not found.")
        return

    kiwi, ppl_bundle, visual_bundle = load_models()
    
    print("\n📖 Reading Data...")
    with open(BASELINE_FILE, 'r') as f: df_base = pd.DataFrame(json.load(f))
    with open(FINETUNED_FILE, 'r') as f: df_tune = pd.DataFrame(json.load(f))
    
    results = []
    all_langs = sorted(df_base['language'].unique().tolist())
    
    def add_row(lang, model_name, metrics):
        results.append({"Language": lang, "Model": model_name, "PPL": metrics[0], "BERTScore": metrics[1], "Kiwi": metrics[2], "SigLIP": metrics[3], "CLIP-B": metrics[4], "CLIP-L": metrics[5]})

    print(f"\n📊 Processing [GLOBAL AVERAGE] ...")
    b = compute_metrics(df_base, kiwi, ppl_bundle, visual_bundle)
    t = compute_metrics(df_tune, kiwi, ppl_bundle, visual_bundle)
    add_row("AVERAGE", "Baseline", b)
    add_row("", "Ours", t)

    for lang in tqdm(all_langs, desc="Processing"):
        sub_base = df_base[df_base['language'] == lang]
        sub_tune = df_tune[df_tune['language'] == lang]
        if len(sub_base) == 0: continue
        b = compute_metrics(sub_base, kiwi, ppl_bundle, visual_bundle)
        t = compute_metrics(sub_tune, kiwi, ppl_bundle, visual_bundle)
        add_row(lang, "Baseline", b)
        add_row("", "Ours", t)

    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 1200)
    print("\n" + "="*120)
    print("🏆 FINAL RESULTS 🏆")
    print("="*120)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()