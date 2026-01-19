import os
import json
import sys
import argparse
import torch
import numpy as np
import random
from PIL import Image, ImageFile
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ✅ 核心导入：新增 CLIPModel, CLIPProcessor
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoProcessor,
    CLIPModel,
    CLIPProcessor
)

# 防止图片截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 开启 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 用户配置区域
# ==========================================
DEFAULT_INTERNVL_PATH = "/mnt/raid/hss/model/InternVL3-8B"
DEFAULT_SIGLIP_PATH = "/mnt/raid/hss/model/siglip-so400m-patch14-384"
# ✅ 新增：本地 CLIP 模型路径 (请修改为你实际的本地路径)
DEFAULT_CLIP_PATH = "/mnt/raid/hss/model/CLIP-ViT-H-14"
DEFAULT_INPUT_DIR = "/mnt/raid/hss/dataset/Image50K"
DEFAULT_FEATURE_PATH = "./siglip_features.pt"

# ==========================================
# InternVL 官方图像处理逻辑 (保持不变)
# ==========================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==========================================
# 辅助函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text):
    if not text: return ""
    text = text.strip()
    special_tokens = ["<s>", "</s>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    for t in special_tokens:
        text = text.replace(t, "")
    lower_text = text.lower()
    prefixes = ["the image shows", "this image shows", "an image of", "depicted here is", "caption:", "output:"]
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            if text.startswith(":") or text.startswith(","): text = text[1:].strip()
            break
    if len(text) > 0: text = text[0].upper() + text[1:]
    return text.strip()

def load_clip_model(model_path, device):
    """
    ✅ 加载本地 CLIP 模型
    """
    print(f"[{device}] Loading CLIP: {model_path} ...", file=sys.stderr)
    try:
        model = CLIPModel.from_pretrained(model_path, device_map=device, local_files_only=True).eval()
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        return model, processor
    except Exception as e:
        print(f"❌ CLIP Load Error: {e}", file=sys.stderr)
        sys.exit(1)

@torch.no_grad()
def rank_and_score(img_path, img_id, img_features_db, candidates, 
                   siglip_model, siglip_processor, 
                   clip_model, clip_processor, device):
    """
    ✅ 综合评分函数：
    1. SigLIP: 用于排序 (Ranking)
    2. CLIP: 用于评分 (Scoring)
    """
    candidates = list(set(candidates)) # 去重
    if not candidates: return "", 0.0, 0.0, []
    
    # -------------------------------------------------------
    # 1. SigLIP 排序逻辑 (Ranking) - 使用预计算特征
    # -------------------------------------------------------
    img_feat_siglip = None
    keys_to_try = [
        img_id, 
        os.path.basename(img_id),
        os.path.splitext(os.path.basename(img_id))[0]
    ]
    
    # 尝试从库中获取图像特征
    if img_features_db is not None:
        for k in keys_to_try:
            if k in img_features_db:
                img_feat_siglip = img_features_db[k]
                break
    
    # 默认分数
    siglip_probs = [0.0] * len(candidates)

    if img_feat_siglip is not None:
        img_feat_siglip = img_feat_siglip.to(device)
        img_feat_siglip = img_feat_siglip / img_feat_siglip.norm(p=2, dim=-1, keepdim=True)

        try:
            text_inputs = siglip_processor(
                text=candidates, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            text_feats = siglip_model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
            
            logit_scale = siglip_model.logit_scale.exp()
            logits = torch.matmul(img_feat_siglip, text_feats.t()) * logit_scale
            
            probs = torch.sigmoid(logits)
            if probs.dim() == 0: probs = [probs.item()]
            else: probs = probs.squeeze().cpu().tolist()
            if isinstance(probs, float): probs = [probs]
            
            siglip_probs = probs
        except Exception as e:
            print(f"SigLIP Ranking Error: {e}", file=sys.stderr)

    # -------------------------------------------------------
    # 2. CLIP Score 计算逻辑 (Scoring) - 实时计算
    # -------------------------------------------------------
    clip_scores = [0.0] * len(candidates)
    try:
        # 读取原图用于 CLIP 计算
        image = Image.open(img_path).convert("RGB")
        
        inputs = clip_processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77 
        ).to(device)
        
        outputs = clip_model(**inputs)
        
        # 计算余弦相似度
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # [1, D] @ [N, D].T = [1, N]
        cosine_sim = torch.matmul(img_embeds, text_embeds.t()).squeeze(0).cpu().tolist()
        if isinstance(cosine_sim, float): cosine_sim = [cosine_sim]
        
        # 保留4位小数
        clip_scores = [round(s, 4) for s in cosine_sim]

    except Exception as e:
        # 容错：如果找不到图或出错，分为0
        pass

    # -------------------------------------------------------
    # 3. 结果合并与排序
    # -------------------------------------------------------
    combined = []
    for i, text in enumerate(candidates):
        combined.append({
            "text": text,
            "score": round(siglip_probs[i], 4), # SigLIP 分数 (用于兼容旧字段名)
            "siglip_score": round(siglip_probs[i], 4),
            "clip_score": clip_scores[i]
        })
    
    # 依然依据 SigLIP 分数选出最好的 Caption (Ranking)
    combined.sort(key=lambda x: x['siglip_score'], reverse=True)
    
    best_candidate = combined[0]
    
    return (
        best_candidate['text'], 
        best_candidate['siglip_score'], 
        best_candidate['clip_score'], 
        combined
    )

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="InternVL3 Offline Batch Generation & CLIPScore")
    
    # 核心参数
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--feature_file", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    # 路径参数
    parser.add_argument("--internvl_path", type=str, default=DEFAULT_INTERNVL_PATH)
    parser.add_argument("--siglip_path", type=str, default=DEFAULT_SIGLIP_PATH)
    # ✅ 新增 CLIP 参数
    parser.add_argument("--clip_path", type=str, default=DEFAULT_CLIP_PATH, help="本地 CLIP 模型路径")
    parser.add_argument("--image_root", type=str, default=DEFAULT_INPUT_DIR)
    
    # 控制参数
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()
    
    set_seed(42 + args.gpu_id)
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    print(f"[{args.gpu_id}] Initializing on {device_str}...", file=sys.stderr)

    # 1. 加载 SigLIP 特征库
    image_features_db = None
    if args.feature_file and os.path.exists(args.feature_file):
        print(f"[{args.gpu_id}] Loading Features...", file=sys.stderr)
        try:
            image_features_db = torch.load(args.feature_file, map_location="cpu")
        except Exception as e:
            print(f"[{args.gpu_id}] Feature load failed: {e}", file=sys.stderr)

    # 2. 加载 InternVL 模型
    print(f"[{args.gpu_id}] Loading InternVL...", file=sys.stderr)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.internvl_path, trust_remote_code=True, use_fast=False, local_files_only=True)
        model = AutoModel.from_pretrained(
            args.internvl_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        ).eval().cuda()
    except Exception as e:
        print(f"InternVL Critical Load Error: {e}")
        return

    # 3. 加载 SigLIP 模型
    siglip_model, siglip_processor = None, None
    if image_features_db is not None:
        try:
            print(f"[{args.gpu_id}] Loading SigLIP...", file=sys.stderr)
            siglip_model = AutoModel.from_pretrained(args.siglip_path, local_files_only=True).to(device_str).eval()
            siglip_processor = AutoProcessor.from_pretrained(args.siglip_path, local_files_only=True)
        except Exception as e:
            print(f"SigLIP Load Error: {e}", file=sys.stderr)
            image_features_db = None 

    # ✅ 4. 加载 CLIP 模型 (本地)
    clip_model, clip_processor = load_clip_model(args.clip_path, device_str)

    # 5. 数据读取
    all_data = []
    try:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content: return
            if content.startswith('['): all_data = json.loads(content)
            else: all_data = [json.loads(line) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"Manifest Error: {e}"); return

    start = max(0, args.start_idx)
    end = len(all_data) if args.end_idx == -1 else min(args.end_idx, len(all_data))
    subset_data = all_data[start:end]
    print(f"[{args.gpu_id}] Task Range: {start} to {end}", file=sys.stderr)

    # 6. 处理循环
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['image_id'])
                except: pass

    f_out = open(args.output_file, 'a', encoding='utf-8')
    
    prompt_short = "Describe the main objects in the image in English within 18 words. Focus strictly on visual attributes. Do not describe text or watermarks."
    prompt_long = "Describe the image in detail in English within 45 words. Focus on appearance and actions. In a second sentence, describe spatial relationships. Ignore watermarks."
    gen_config = dict(max_new_tokens=128, do_sample=True, temperature=0.8, top_p=0.9)

    for item in tqdm(subset_data, desc=f"GPU{args.gpu_id}"):
        img_id = item.get('image_id', 'unknown')
        if img_id in processed_ids: continue

        # 路径查找
        img_path = item.get('path', '')
        if not os.path.exists(img_path):
            alt = os.path.join(args.image_root, os.path.basename(img_path))
            if os.path.exists(alt): img_path = alt
            else: continue

        try:
            # 加载图像 (InternVL 格式)
            pixel_values = None
            with Image.open(img_path) as img:
                raw_image = img.convert("RGB")
                width, height = raw_image.size
                pixel_values = load_image(raw_image, max_num=12).to(model_dtype).to(device_str)

            # 生成
            short_cands = []
            for _ in range(args.num_candidates):
                resp = model.chat(tokenizer, pixel_values, prompt_short, gen_config)
                short_cands.append(clean_text(resp))

            long_cands = []
            for _ in range(args.num_candidates):
                resp = model.chat(tokenizer, pixel_values, prompt_long, gen_config)
                long_cands.append(clean_text(resp))

            # ✅ 评分与排序 (传入 img_path 和 CLIP 模型)
            best_short, s_score, s_clip, s_details = rank_and_score(
                img_path, img_id, image_features_db, short_cands, 
                siglip_model, siglip_processor,
                clip_model, clip_processor, 
                device_str
            )
            
            best_long, l_score, l_clip, l_details = rank_and_score(
                img_path, img_id, image_features_db, long_cands, 
                siglip_model, siglip_processor,
                clip_model, clip_processor, 
                device_str
            )

            # 写入结果
            res = {
                "image_id": img_id,
                "path": img_path,
                "width": width, "height": height,
                # Short
                "short_caption": best_short, # 兼容旧字段
                "short_caption_best": best_short,
                "short_score": s_score,
                "short_clip_score": s_clip, # ✅ CLIP 分数
                "short_candidates": s_details,
                # Long
                "long_caption": best_long,   # 兼容旧字段
                "long_caption_best": best_long,
                "long_score": l_score,
                "long_clip_score": l_clip, # ✅ CLIP 分数
                "long_candidates": l_details
            }
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush()

            if len(long_cands) > 5: torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on {img_id}: {e}", file=sys.stderr)
            torch.cuda.empty_cache()
            continue

    f_out.close()
    print(f"[{args.gpu_id}] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()