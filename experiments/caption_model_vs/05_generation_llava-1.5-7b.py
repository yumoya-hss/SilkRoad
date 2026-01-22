import os
import json
import sys
import argparse
import torch
import numpy as np
import random
from PIL import Image, ImageFile
from tqdm import tqdm
import warnings

# 忽略非致命警告
warnings.filterwarnings("ignore")

# ✅ 核心导入：新增 CLIPModel, CLIPProcessor
from transformers import (
    LlavaForConditionalGeneration, # LLaVA-1.5 使用这个类
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
DEFAULT_LLAVA_PATH = "models/llava-1.5-7b-hf"
DEFAULT_SIGLIP_PATH = "models/siglip-so400m-patch14-384"
# ✅ 新增：本地 CLIP 模型路径 (请修改为你实际的本地路径)
DEFAULT_CLIP_PATH = "models/CLIP-ViT-H-14"
DEFAULT_INPUT_DIR = "data/Image50K"
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

def load_clip_model(model_path, device):
    """
    ✅ 加载本地 CLIP 模型用于评分
    """
    print(f"[{device}] 正在加载本地 CLIP 模型: {model_path} ...", file=sys.stderr)
    try:
        # local_files_only=True 确保不联网
        model = CLIPModel.from_pretrained(model_path, device_map=device, local_files_only=True).eval()
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        return model, processor
    except Exception as e:
        print(f"❌ CLIP 加载失败: {e}", file=sys.stderr)
        sys.exit(1)

def clean_text(text):
    if not text: return ""
    text = text.strip()
    
    # 移除特殊 Token
    special_tokens = ["<s>", "</s>", "ASSISTANT:", "USER:", "<image>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    text = text.strip()
    lower_text = text.lower()
    
    prefixes = [
        "the image shows", "this image shows", "an image of", 
        "depicted here is", "in this image,", "here we can see",
        "the main object in this image is"
    ]
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            if text and text[0] in [":", ",", "."]: 
                text = text[1:].strip()
            break
            
    if len(text) > 0: 
        text = text[0].upper() + text[1:]
    return text

@torch.no_grad()
def rank_and_score(img_path, img_id, img_features_db, candidates, 
                   siglip_model, siglip_processor, 
                   clip_model, clip_processor, device):
    """
    ✅ 综合评分函数：
    1. SigLIP: 用于排序 (Ranking)
    2. CLIP: 用于评分 (Scoring)
    """
    if not candidates: return "", 0.0, 0.0, []
    
    # -------------------------------------------------------
    # 1. SigLIP 排序逻辑 (Ranking) - 使用预计算特征
    # -------------------------------------------------------
    img_feat_siglip = None
    keys_to_try = [
        img_id, 
        os.path.splitext(img_id)[0], 
        os.path.basename(img_id),
        os.path.splitext(os.path.basename(img_id))[0]
    ]
    
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
            
            logit_scale = siglip_model.logit_scale.exp()
            logit_bias = getattr(siglip_model, 'logit_bias', 0)
            
            logits = (torch.matmul(img_feat_siglip, text_feats.t()) * logit_scale) + logit_bias
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
            if isinstance(probs, float): probs = [probs]
            siglip_probs = probs
        except Exception as e:
            print(f"SigLIP Ranking Error: {e}", file=sys.stderr)

    # -------------------------------------------------------
    # 2. CLIP Score 计算逻辑 (Scoring) - 实时计算
    # -------------------------------------------------------
    clip_scores = [0.0] * len(candidates)
    try:
        # 读取图像用于 CLIP 计算
        image = Image.open(img_path).convert("RGB")
        
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
    # 3. 结果合并与排序
    # -------------------------------------------------------
    combined = []
    for i, text in enumerate(candidates):
        combined.append({
            "text": text,
            "siglip_score": round(siglip_probs[i], 4),
            "clip_score": clip_scores[i]
        })
    
    # 依然依据 SigLIP 分数选出最好的 (保持原有逻辑)
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
    parser = argparse.ArgumentParser(description="LLaVA-1.5 Pipeline with CLIPScore")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--llava_path", type=str, default=DEFAULT_LLAVA_PATH)
    parser.add_argument("--siglip_path", type=str, default=DEFAULT_SIGLIP_PATH)
    # ✅ 新增 CLIP 参数
    parser.add_argument("--clip_path", type=str, default=DEFAULT_CLIP_PATH, help="本地 CLIP 模型路径")
    parser.add_argument("--image_root", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--feature_file", type=str, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()
    set_seed(42 + args.gpu_id)
    
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu_id)

    # 1. 加载特征库 (SigLIP)
    image_features_db = None
    if args.feature_file and os.path.exists(args.feature_file):
        print(f"[{args.gpu_id}] 加载图像特征库...", file=sys.stderr)
        try:
            image_features_db = torch.load(args.feature_file, map_location="cpu")
        except Exception as e:
            print(f"特征加载失败 ({e})，跳过排序。", file=sys.stderr)
            image_features_db = {}
    else:
        print("未找到特征文件，跳过排序。", file=sys.stderr)
        image_features_db = {}

    # 2. 读取数据
    print(f"[{args.gpu_id}] 读取 Manifest...", file=sys.stderr)
    all_data = []
    try:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                f.seek(0); all_data = json.load(f)
            else:
                for line in content.split('\n'):
                    if line.strip(): all_data.append(json.loads(line))
    except Exception as e:
        print(f"Manifest 读取错误: {e}", file=sys.stderr); return

    start = max(0, args.start_idx)
    end = len(all_data) if args.end_idx == -1 else min(args.end_idx, len(all_data))
    subset_data = all_data[start:end]
    print(f"[{args.gpu_id}] 任务量: {len(subset_data)} 条", file=sys.stderr)

    # 3. 加载 LLaVA-1.5 模型
    print(f"[{args.gpu_id}] 加载 LLaVA-1.5 ({args.llava_path})...", file=sys.stderr)
    
    attn_implementation = "sdpa"
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("✅ Flash Attention 2 Enabled.", file=sys.stderr)
    except ImportError:
        print("⚠️ SDPA Enabled.", file=sys.stderr)

    try:
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            args.llava_path, 
            torch_dtype=torch.float16, 
            attn_implementation=attn_implementation,
            device_map=device_str,
            trust_remote_code=True
        ).eval()
        
        llava_processor = AutoProcessor.from_pretrained(args.llava_path, trust_remote_code=True)
        
        if llava_processor.tokenizer.pad_token_id is None:
            llava_processor.tokenizer.pad_token_id = llava_processor.tokenizer.eos_token_id
            
    except Exception as e:
        print(f"❌ LLaVA-1.5 加载失败: {e}", file=sys.stderr)
        print("提示: 确保 transformers 版本 >= 4.36.0", file=sys.stderr)
        return

    # 4. 加载 SigLIP (Text Encoder)
    siglip_model, siglip_processor = None, None
    print(f"[{args.gpu_id}] 加载 SigLIP...", file=sys.stderr)
    try:
        siglip_model = AutoModel.from_pretrained(args.siglip_path, device_map=device_str).eval()
        siglip_processor = AutoProcessor.from_pretrained(args.siglip_path)
    except Exception as e:
        print(f"⚠️ SigLIP 加载失败: {e}", file=sys.stderr)

    # ✅ 5. 加载 CLIP 模型 (本地)
    clip_model, clip_processor = load_clip_model(args.clip_path, device_str)

    # 6. 生成闭包
    def generate_batch(messages, num_return):
        prompts = []
        for msg in messages:
            try:
                prompt = llava_processor.apply_chat_template(msg, add_generation_prompt=True)
            except Exception:
                # Fallback: USER: <image>\nPrompt\nASSISTANT:
                user_content = msg[0]['content']
                text_query = user_content[1]['text']
                prompt = f"USER: <image>\n{text_query}\nASSISTANT:"
            prompts.append(prompt)
        
        image_inputs = [msg[0]["content"][0]["image"] for msg in messages]

        inputs = llava_processor(
            text=prompts, 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(device_str)

        with torch.inference_mode():
            user_text = messages[0][0]['content'][1]['text']
            max_new = 64 if "within 18 words" in user_text else 128
            
            generated_ids = llava_model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                num_return_sequences=num_return,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=llava_processor.tokenizer.pad_token_id 
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        raw_texts = llava_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [clean_text(t) for t in raw_texts]

    # 7. 处理循环
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
                with Image.open(full_img_path) as img:
                    raw_image = img.convert("RGB")
                    
                    msgs_short = [{"role": "user", "content": [{"type": "image", "image": raw_image}, 
                        {"type": "text", "text": "Describe the main objects in the image in English within 18 words. Focus strictly on visual attributes. Do not describe text or watermarks."}]}]
                    
                    msgs_long = [{"role": "user", "content": [{"type": "image", "image": raw_image}, 
                        {"type": "text", "text": "Describe the image in detail in English within 45 words. Focus on appearance and actions. In a second sentence, describe spatial relationships. Ignore watermarks."}]}]

                    # 生成
                    short_cands = generate_batch([msgs_short], args.num_candidates)
                    long_cands = generate_batch([msgs_long], args.num_candidates)

            except Exception as e:
                print(f"Gen Error {img_id}: {e}", file=sys.stderr)
                torch.cuda.empty_cache()
                continue

            try:
                # ✅ 评分与排序 (传入 CLIP 模型)
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
                    "width": raw_image.size[0], "height": raw_image.size[1],
                    "short_caption_best": best_short,
                    "short_score": s_siglip,
                    "short_clip_score": s_clip, # ✅ CLIP 分数
                    "short_candidates": s_details,
                    "long_caption_best": best_long,
                    "long_score": l_siglip,
                    "long_clip_score": l_clip, # ✅ CLIP 分数
                    "long_candidates": l_details
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"Rank/Save Error {img_id}: {e}", file=sys.stderr)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
