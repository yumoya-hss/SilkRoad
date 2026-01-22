import json
import sys
import os
import torch
import argparse
import gc
import numpy as np
import traceback
import yaml 
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel, XLMRobertaTokenizer # 必须导入这个用于手动加载分词器

# ==========================================
# 🔥 强制离线模式 (最优先设置) 🔥
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 延迟导入评测库
from bert_score import score as bert_score_func
from comet import download_model, load_from_checkpoint
# 导入 COMET 内部模块用于拦截
from comet.encoders import xlmr 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 🔥 [配置区域] (硬编码路径) 🔥
# ==========================================

# 1. 输入输出
# 上一步回译(04_b)的输出文件
DEFAULT_INPUT_FILE = "dataset_with_bt.json"
# 最终带分数的输出文件
DEFAULT_OUTPUT_FILE = "dataset_scored_final.json"

# 2. 模型路径
# COMET-Kiwi (SOTA QE模型): 自动下载或指定本地路径
# 如果服务器有网，直接填 "Unbabel/wmt22-cometkiwi-da"
# 离线环境请指向本地路径 (代码会自动寻找 .ckpt)
DEFAULT_COMET_PATH = "models/wmt22-cometkiwi-da" 

# 🔥 [新增] COMET 底座模型路径 (必须手动下载 infoxlm-large) 🔥
# 用于防止 COMET 尝试联网下载 microsoft/infoxlm-large
DEFAULT_COMET_ENCODER_PATH = "models/infoxlm-large"

# CLIP 模型 (用于视觉一致性)
DEFAULT_CLIP_PATH = "models/clip-vit-large-patch14"

# 🔥 [新增] BERTScore 模型 (本地路径) 🔥
# 您下载的是 xlm-roberta-large，这是一个强大的多语言模型
DEFAULT_BERT_PATH = "models/xlm-roberta-large"

# 图片根目录 (路径回退用)
DEFAULT_IMAGE_ROOT = "data/Image50K"

# 3. 批次大小 (根据显存优化)
BATCH_SIZE_COMET = 32
BATCH_SIZE_BERT = 64
BATCH_SIZE_CLIP = 64

# GPU ID
DEFAULT_GPU_ID = 0

# ==========================================

def load_data(file_path):
    print(f"📖 读取数据: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 兼容 JSON Array
            if f.read(1) == '[':
                f.seek(0)
                return json.load(f)
            # 兼容 JSONL
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        sys.exit(1)

def save_data(data, path):
    print(f"💾 保存结果至: {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================================
# 🔥 [核心修复模块] 内存级拦截 (Double Monkey Patch) 🔥
# =========================================================================

def ensure_config_exists(encoder_path):
    """
    确保本地目录下有 config.json，否则加载本地路径也会报错。
    如果只有 sentencepiece.bpe.model 而没有 config.json，则自动生成一个。
    """
    config_path = os.path.join(encoder_path, "config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ 警告: {config_path} 不存在，正在生成标准 InfoXLM 配置...")
        # 标准 InfoXLM-large 配置
        config_data = {
            "architectures": ["XLMRobertaForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "model_type": "xlm-roberta",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 250002,
            "sentencepiece_model_file": "sentencepiece.bpe.model",
            "tokenizer_class": "XLMRobertaTokenizer"
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print("✅ config.json 生成完毕。")
        except Exception as e:
            print(f"❌ 生成配置文件失败: {e}")

def apply_monkey_patches(local_encoder_path):
    """
    🔥 核心：双重拦截 (Tokenizer + Encoder) 🔥
    直接修改 COMET 源代码在内存中的行为，强制使用我们指定的本地路径。
    """
    print(f"🔧 [Monkey Patch] 正在注入拦截逻辑 (Override transformers loading)...")
    
    # -------------------------------------------
    # 1. Tokenizer 拦截 (解决 OSError: Not found: "None")
    # -------------------------------------------
    vocab_file = os.path.join(local_encoder_path, "sentencepiece.bpe.model")
    if not os.path.exists(vocab_file):
        # 尝试递归查找
        possible_vocabs = [os.path.join(r, f) for r, d, f in os.walk(local_encoder_path) if f == "sentencepiece.bpe.model"]
        if possible_vocabs:
            vocab_file = possible_vocabs[0]
        else:
            raise FileNotFoundError(f"❌ 致命错误: 在 {local_encoder_path} 中找不到 sentencepiece.bpe.model")
    
    print(f"    ⏳ 手动加载 Tokenizer (from {vocab_file})...")
    # 手动加载好的对象 (Slow Tokenizer)
    my_tokenizer = XLMRobertaTokenizer(vocab_file=vocab_file)
    
    # 拦截 from_pretrained，永远返回我们手动加载的
    def fake_tokenizer_loader(*args, **kwargs):
        print("    🛡️  [Tokenizer] 拦截成功，直接返回预加载的分词器对象。")
        return my_tokenizer
    
    # 覆盖类方法
    xlmr.XLMRobertaTokenizerFast.from_pretrained = fake_tokenizer_loader

    # -------------------------------------------
    # 2. Encoder 拦截 (解决 MaxRetryError 联网报错)
    # -------------------------------------------
    # 保存原始方法，因为我们需要调用它，只是改个参数
    original_encoder_loader = xlmr.XLMREncoder.from_pretrained

    def fake_encoder_loader(pretrained_model, load_pretrained_weights=True, local_files_only=False):
        print(f"    🛡️  [Encoder] 拦截模型加载请求: '{pretrained_model}'")
        print(f"    ➡️  强制重定向到本地路径: '{local_encoder_path}'")
        
        # 强制把第一个参数(模型名)改成我们的本地绝对路径
        # 这样 transformers 就会去本地找 config.json，而不再联网
        return original_encoder_loader(local_encoder_path, load_pretrained_weights, local_files_only=True)

    # 覆盖类方法
    xlmr.XLMREncoder.from_pretrained = fake_encoder_loader
    
    print("✅ 双重补丁注入完成！COMET 现在将完全离线运行。")

# =========================================================================

def run_scoring(data, args):
    device = f"cuda:{args.gpu_id}"
    print(f"🚀 开始评分流程 (Device: {device})...")

    # ------------------------------------------------------
    # 1. BERTScore (Text Consistency: English Source vs BackTrans)
    # ------------------------------------------------------
    print(f"\n[1/3] 计算 BERTScore (语义一致性)...")
    print(f"      Loading Local BERT: {args.bert_path}")
    
    cands = [] # 候选: 回译文本
    refs = []  # 参考: 原始英文Caption
    map_indices = [] # 记录索引以便回填: (data_idx, lang, key)

    # 遍历数据收集 Batch
    for idx, item in enumerate(data):
        if 'translations' not in item: continue
        
        # 英文原句 (Source)
        short_src = item.get('short_caption_best', '')
        long_src = item.get('long_caption_best', '')

        for lang, trans_obj in item['translations'].items():
            # 动态遍历所有键值，自动适配维吾尔语(只有NLLB)和其他语言(双模型)
            for key, val in trans_obj.items():
                # 我们只关心 'bt_' 开头的回译字段
                if not key.startswith('bt_'): continue
                
                back_trans = val
                if not back_trans: continue # 跳过空值

                # 确定对应的原句是 Short 还是 Long
                # key 格式如: bt_short_nllb, bt_long_seamless
                ref_text = short_src if 'short' in key else long_src
                if not ref_text: continue

                cands.append(back_trans)
                refs.append(ref_text)
                # 记录: 原始key去掉 'bt_' 就是翻译key, 例如 short_nllb
                original_trans_key = key[3:] 
                map_indices.append((idx, lang, original_trans_key))

    # 执行 BERTScore 计算
    if cands:
        try:
            # 🔥 关键修改: 去掉 lang="en"，只使用 model_type 指定本地路径
            # xlm-roberta-large 也是 24 层，num_layers=17 是一个经验值
            P, R, F1 = bert_score_func(
                cands, 
                refs, 
                model_type=args.bert_path, # 强制使用本地路径
                num_layers=17,             # 保持默认层数选择
                verbose=True, 
                device=device, 
                batch_size=args.batch_size_bert
            )
            
            # 回填分数
            for i, (idx, lang, key) in enumerate(map_indices):
                # 字段名: score_bert_short_nllb
                score_key = f"score_bert_{key}"
                data[idx]['translations'][lang][score_key] = float(F1[i])
        except Exception as e:
            print(f"❌ BERTScore 计算出错: {e}")
            print("请检查 DEFAULT_BERT_PATH 是否正确指向了包含 config.json 的文件夹。")
            traceback.print_exc()
            
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------
    # 2. COMET-Kiwi (Translation Quality: English Source vs Target Translation)
    # ------------------------------------------------------
    print(f"\n[2/3] 计算 COMET-Kiwi (无参考翻译质量)...")
    print(f"      Loading Model Dir: {args.comet_path}")
    
    # 🔥 [应用修复] 1. 确保 config 存在 (防止本地加载报错) 🔥
    ensure_config_exists(args.comet_encoder_path)
    # 🔥 [应用修复] 2. 注入 Monkey Patch (接管加载过程) 🔥
    apply_monkey_patches(args.comet_encoder_path)
    
    comet_model = None
    try:
        # 🔥 第三步：自动寻找 .ckpt 文件 🔥
        # load_from_checkpoint 必须指向文件，不能指向文件夹
        ckpt_path = args.comet_path
        if os.path.isdir(ckpt_path):
            print("      (Detecting checkpoint in directory...)")
            # 优先找 checkpoints/model.ckpt (标准结构)
            possible_ckpt = os.path.join(ckpt_path, "checkpoints", "model.ckpt")
            if os.path.exists(possible_ckpt):
                ckpt_path = possible_ckpt
            else:
                # 否则搜索目录下任何 .ckpt 文件
                found = False
                for root, dirs, files in os.walk(ckpt_path):
                    for f in files:
                        if f.endswith(".ckpt"):
                            ckpt_path = os.path.join(root, f)
                            found = True
                            break
                    if found: break
        
        print(f"      Target Checkpoint File: {ckpt_path}")
        
        # 🔥 第四步：标准加载 🔥
        # 此时所有的 from_pretrained 调用都会被我们的 patch 拦截并重定向
        comet_model = load_from_checkpoint(ckpt_path).to(device).eval()
        print("✅ COMET 模型加载成功！")
        
    except Exception as e:
        print(f"❌ COMET 加载失败: {e}")
        traceback.print_exc()
        print("无法加载 COMET 模型，跳过此步骤。")
        comet_model = None

    if comet_model:
        comet_data = [] # [{"src": "...", "mt": "..."}]
        comet_indices = []

        for idx, item in enumerate(data):
            if 'translations' not in item: continue
            
            short_src = item.get('short_caption_best', '')
            long_src = item.get('long_caption_best', '')

            for lang, trans_obj in item['translations'].items():
                for key, val in trans_obj.items():
                    # 跳过回译(bt_)和已有的分数(score_)
                    if key.startswith('bt_') or key.startswith('score_'): continue
                    
                    translation = val
                    if not translation: continue

                    src_text = short_src if 'short' in key else long_src
                    if not src_text: continue

                    # COMET-Kiwi 输入: 源语言(英文) + 目标语言译文
                    comet_data.append({"src": src_text, "mt": translation})
                    comet_indices.append((idx, lang, key))

        if comet_data:
            print(f"      Running prediction on {len(comet_data)} samples...")
            try:
                model_output = comet_model.predict(comet_data, batch_size=args.batch_size_comet, gpus=1)
                scores = model_output.scores
                
                for i, (idx, lang, key) in enumerate(comet_indices):
                    score_key = f"score_comet_{key}"
                    data[idx]['translations'][lang][score_key] = float(scores[i])
            except Exception as e:
                print(f"❌ COMET 推理出错: {e}")
                traceback.print_exc()

        del comet_model
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------
    # 3. Visual Consistency (CLIP Score: Image vs English BackTrans)
    # ------------------------------------------------------
    print(f"\n[3/3] 计算 CLIP Score (视觉一致性)...")
    print(f"      Loading CLIP: {args.clip_path}")
    
    try:
        clip_model = CLIPModel.from_pretrained(args.clip_path).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    except Exception as e:
        print(f"❌ CLIP 加载失败: {e}")
        clip_model = None

    if clip_model:
        # CLIP 只能按图处理，因为每张图对应多个文本
        for idx, item in tqdm(enumerate(data), total=len(data), desc="CLIP Scoring"):
            img_path = item.get('path', '')
            
            # 路径检查与回退
            if not os.path.exists(img_path):
                filename = os.path.basename(img_path)
                # 尝试去默认图片目录找
                fallback_path = os.path.join(args.image_root, filename)
                if os.path.exists(fallback_path):
                    img_path = fallback_path
                else:
                    continue

            try:
                image = Image.open(img_path).convert("RGB")
            except: continue

            # 收集这张图的所有回译文本
            texts = []
            keys_map = [] # (lang, original_key)

            if 'translations' not in item: continue
            for lang, trans_obj in item['translations'].items():
                for key, val in trans_obj.items():
                    if key.startswith('bt_') and val:
                        # CLIP 文本长度限制 77 token，做个截断防止报错
                        texts.append(val[:77]) 
                        original_key = key[3:] # 去掉 bt_
                        keys_map.append((lang, original_key))
            
            if not texts: continue

            # 推理
            inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                # 计算余弦相似度
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                # [1, embed_dim] @ [embed_dim, num_texts] -> [1, num_texts]
                cosine_scores = (image_embeds @ text_embeds.t()).squeeze(0).cpu().numpy()
                
                if isinstance(cosine_scores, float): cosine_scores = [cosine_scores]

            # 回填
            for i, (lang, original_key) in enumerate(keys_map):
                score_key = f"score_visual_{original_key}"
                item['translations'][lang][score_key] = float(cosine_scores[i])

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    
    # 路径参数
    parser.add_argument("--comet_path", type=str, default=DEFAULT_COMET_PATH)
    # 新增 Encoder 参数
    parser.add_argument("--comet_encoder_path", type=str, default=DEFAULT_COMET_ENCODER_PATH) 
    parser.add_argument("--clip_path", type=str, default=DEFAULT_CLIP_PATH)
    parser.add_argument("--bert_path", type=str, default=DEFAULT_BERT_PATH) # 新增
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    
    # 显存控制
    parser.add_argument("--batch_size_comet", type=int, default=BATCH_SIZE_COMET)
    parser.add_argument("--batch_size_bert", type=int, default=BATCH_SIZE_BERT)
    parser.add_argument("--batch_size_clip", type=int, default=BATCH_SIZE_CLIP)
    
    # GPU 控制
    parser.add_argument("--gpu_id", type=int, default=DEFAULT_GPU_ID)
    
    args = parser.parse_args()

    # 1. 加载
    dataset = load_data(args.input_file)
    
    # 2. 评分 (包含所有逻辑)
    dataset = run_scoring(dataset, args)
    
    # 3. 保存
    save_data(dataset, args.output_file)
