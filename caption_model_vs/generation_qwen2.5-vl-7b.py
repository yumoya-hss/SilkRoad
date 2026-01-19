import os
import json
import sys
import argparse
import torch
import numpy as np
import random
from PIL import Image, ImageFile
from tqdm import tqdm

# ✅ 核心修正：导入 CLIP 相关的类
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    AutoModel, 
    CLIPModel, 
    CLIPProcessor
)

# 防止图片截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 用户配置区域
# ==========================================
DEFAULT_QWEN_PATH = "/mnt/raid/hss/model/Qwen2.5-VL-7B-Instruct"
DEFAULT_SIGLIP_PATH = "/mnt/raid/hss/model/siglip-so400m-patch14-384"
# ✅ 新增：本地 CLIP 模型路径 (请确保此路径下有 config.json, pytorch_model.bin 等文件)
DEFAULT_CLIP_PATH = "/mnt/raid/hss/model/CLIP-ViT-H-14" 
DEFAULT_INPUT_DIR = "/mnt/raid/hss/dataset/Image50K"
DEFAULT_FEATURE_PATH = "./siglip_features.pt" 

# ==========================================
# 辅助函数
# ==========================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_siglip_text_only(model_path, device):
    """
    只加载 SigLIP 用于文本编码，节省显存 (配合预提取的图像特征使用)
    """
    print(f"[{device}] 正在加载 SigLIP (Text Encoder)...", file=sys.stderr)
    try:
        model = AutoModel.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor
    except Exception as e:
        print(f"SigLIP 加载失败: {e}", file=sys.stderr)
        sys.exit(1)

def load_clip_model(model_path, device):
    """
    ✅ 加载本地 CLIP 模型用于计算 CLIPScore
    """
    print(f"[{device}] 正在加载本地 CLIP 模型: {model_path} ...", file=sys.stderr)
    try:
        # local_files_only=True 强制不联网
        model = CLIPModel.from_pretrained(model_path, device_map=device, local_files_only=True).eval()
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        return model, processor
    except Exception as e:
        print(f"CLIP 加载失败: {e}", file=sys.stderr)
        sys.exit(1)

def clean_text(text):
    if not text: return ""
    text = text.strip()
    lower_text = text.lower()
    prefixes = ["the image shows", "this image shows", "an image of", "depicted here is"]
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            if text.startswith(":") or text.startswith(","): text = text[1:].strip()
            break
    if len(text) > 0: text = text[0].upper() + text[1:]
    return text

@torch.no_grad()
def rank_and_score(img_path, img_id, img_features_db, candidates, 
                   siglip_model, siglip_processor, 
                   clip_model, clip_processor, device):
    """
    ✅ 综合评分函数：
    1. SigLIP: 用于排序 (使用缓存特征，速度快)
    2. CLIP: 用于评估 (读取原图计算 Cosine Similarity)
    """
    if not candidates: return "", 0.0, 0.0, []
    
    # -------------------------------------------------------
    # 1. SigLIP 排序逻辑 (Ranking)
    # -------------------------------------------------------
    img_feat_siglip = None
    keys_to_try = [
        img_id, 
        os.path.splitext(img_id)[0], 
        os.path.basename(img_id),
        os.path.splitext(os.path.basename(img_id))[0]
    ]
    
    # 从预计算好的特征库中查找
    for k in keys_to_try:
        if k in img_features_db:
            img_feat_siglip = img_features_db[k]
            break
    
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
            
            # SigLIP Logit 计算
            logit_scale = siglip_model.logit_scale.exp()
            logit_bias = getattr(siglip_model, 'logit_bias', 0)
            
            logits = (torch.matmul(img_feat_siglip, text_feats.t()) * logit_scale) + logit_bias
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
            if isinstance(probs, float): probs = [probs]
            siglip_probs = probs
        except Exception as e:
            print(f"SigLIP Ranking Error: {e}", file=sys.stderr)

    # -------------------------------------------------------
    # 2. CLIP Score 计算逻辑 (Scoring)
    # -------------------------------------------------------
    clip_scores = [0.0] * len(candidates)
    try:
        # CLIP 需要读取原图
        image = Image.open(img_path).convert("RGB")
        
        # 预处理
        inputs = clip_processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77 # CLIP 默认最大长度
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
        print(f"CLIP Scoring Error for {img_id}: {e}", file=sys.stderr)

    # -------------------------------------------------------
    # 3. 合并结果并排序
    # -------------------------------------------------------
    combined = []
    for i, text in enumerate(candidates):
        combined.append({
            "text": text,
            "siglip_score": round(siglip_probs[i], 4), # 用于排序
            "clip_score": clip_scores[i]               # 仅作为参考指标
        })
    
    # 依然依据 SigLIP 分数选出最好的 Caption (Ranking)
    combined.sort(key=lambda x: x['siglip_score'], reverse=True)
    
    best_candidate = combined[0]
    
    return (
        best_candidate['text'], 
        best_candidate['siglip_score'], 
        best_candidate['clip_score'], # 返回最佳候选的 CLIP 分数
        combined # 返回包含 CLIP 分数的详细列表
    )

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Generation & Scoring (SigLIP+CLIP)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--feature_file", type=str, default=DEFAULT_FEATURE_PATH, help="上一步生成的SigLIP特征文件")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    parser.add_argument("--qwen_path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--siglip_path", type=str, default=DEFAULT_SIGLIP_PATH)
    # ✅ 新增参数
    parser.add_argument("--clip_path", type=str, default=DEFAULT_CLIP_PATH, help="本地 CLIP 模型路径")
    parser.add_argument("--image_root", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--num_candidates", type=int, default=5, help="Best-of-N数量")

    args = parser.parse_args()
    set_seed(42 + args.gpu_id)
    
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu_id)

    # 1. 加载预计算特征 (SigLIP, CPU RAM)
    print(f"正在加载图像特征库: {args.feature_file} ...", file=sys.stderr)
    try:
        image_features_db = torch.load(args.feature_file, map_location="cpu")
        print(f"成功加载 {len(image_features_db)} 个图像特征。", file=sys.stderr)
    except Exception as e:
        print(f"特征文件加载失败: {e}", file=sys.stderr)
        image_features_db = {} # 容错，即使没有特征库也继续运行(只生成不排序)

    # 2. 读取 Manifest
    print(f"[{args.gpu_id}] 读取 Manifest...", file=sys.stderr)
    all_data = []
    try:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            if f.read(1) == '[':
                f.seek(0); all_data = json.load(f)
            else:
                f.seek(0)
                for line in f:
                    if line.strip(): all_data.append(json.loads(line))
    except Exception as e:
        print(f"Manifest Error: {e}", file=sys.stderr); return

    # 3. 切片
    start = max(0, args.start_idx)
    end = len(all_data) if args.end_idx == -1 else min(args.end_idx, len(all_data))
    subset_data = all_data[start:end]
    print(f"[{args.gpu_id}] 任务: {len(subset_data)} 条", file=sys.stderr)

    # 4. 加载模型
    print(f"[{args.gpu_id}] 加载 Qwen2.5-VL...", file=sys.stderr)
    try:
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            args.qwen_path, 
            torch_dtype="auto", 
            device_map=device_str, 
            trust_remote_code=True
        ).eval()
        
        qwen_processor = AutoProcessor.from_pretrained(
            args.qwen_path, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Qwen Load Error: {e}", file=sys.stderr)
        return

    print(f"[{args.gpu_id}] 加载 SigLIP (Text Part)...", file=sys.stderr)
    siglip_model, siglip_processor = load_siglip_text_only(args.siglip_path, device_str)

    # ✅ 加载本地 CLIP 模型
    clip_model, clip_processor = load_clip_model(args.clip_path, device_str)

    # 5. 生成闭包
    def generate_batch(messages, num_return):
        text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [msg["content"][0]["image"] for msg in messages]
        inputs = qwen_processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(device_str)
        
        with torch.inference_mode():
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=64 if "within 18 words" in messages[0]['content'][1]['text'] else 128,
                do_sample=True,
                num_return_sequences=num_return,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
        raw_texts = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [clean_text(t) for t in raw_texts]

    # 6. 处理循环
    processed_ids = set()
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                for line in f:
                    try: processed_ids.add(json.loads(line)['image_id'])
                    except: pass
        except: pass

    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        for item in tqdm(subset_data, desc=f"GPU{args.gpu_id}"):
            img_id = item.get('image_id', 'unknown')
            if img_id in processed_ids: continue

            # 路径查找
            full_img_path = item.get('path', '')
            if not os.path.exists(full_img_path):
                filename = os.path.basename(full_img_path)
                candidates = [
                    full_img_path,
                    os.path.join(args.image_root, filename),
                    os.path.join(args.image_root, "imagenet_val", filename)
                ]
                for cand in candidates:
                    if os.path.exists(cand): full_img_path = cand; break
            
            if not os.path.exists(full_img_path): continue

            try:
                # 使用上下文管理器打开图片
                with Image.open(full_img_path) as img:
                    raw_image = img.convert("RGB")
                    width, height = raw_image.size
                    
                    # 定义 Prompt (Blind Captioning)
                    msgs_short = [{"role": "user", "content": [{"type": "image", "image": raw_image}, 
                        {"type": "text", "text": "Describe the main objects in the image in English within 18 words. Focus strictly on visual attributes. Do not describe text or watermarks."}]}]
                    msgs_long = [{"role": "user", "content": [{"type": "image", "image": raw_image}, 
                        {"type": "text", "text": "Describe the image in detail in English within 45 words. Focus on appearance and actions. In a second sentence, describe spatial relationships. Ignore watermarks."}]}]

                    # 生成
                    short_cands = generate_batch(msgs_short, args.num_candidates)
                    long_cands = generate_batch(msgs_long, args.num_candidates)

            except Exception as e:
                print(f"Gen Error {img_id}: {e}", file=sys.stderr)
                torch.cuda.empty_cache()
                continue

            try:
                # ✅ 排序与评分 (SigLIP Ranking + CLIP Scoring)
                # 传入 full_img_path 和 CLIP 模型
                best_short, s_siglip, s_clip, s_details = rank_and_score(
                    full_img_path, img_id, image_features_db, short_cands, 
                    siglip_model, siglip_processor, 
                    clip_model, clip_processor, 
                    device_str
                )
                
                best_long, l_siglip, l_clip, l_details = rank_and_score(
                    full_img_path, img_id, image_features_db, long_cands, 
                    siglip_model, siglip_processor, 
                    clip_model, clip_processor, 
                    device_str
                )

                result = {
                    "image_id": img_id,
                    "path": full_img_path,
                    "wnid": item.get('wnid', ''),
                    "label_name": item.get('label_name', ''),
                    "width": width,
                    "height": height,
                    
                    # Short Results
                    "short_caption_best": best_short,
                    "short_score": s_siglip, # SigLIP 分数
                    "short_clip_score": s_clip, # ✅ 新增 CLIP 分数
                    "short_candidates": s_details,
                    
                    # Long Results
                    "long_caption_best": best_long,
                    "long_score": l_siglip, # SigLIP 分数
                    "long_clip_score": l_clip, # ✅ 新增 CLIP 分数
                    "long_candidates": l_details
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"Rank/Save Error {img_id}: {e}", file=sys.stderr)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
