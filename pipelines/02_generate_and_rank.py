import os
import json
import sys
import argparse
import torch
import numpy as np
import random
from PIL import Image, ImageFile
from tqdm import tqdm
# ✅ 核心修正：导入正确的模型加载类
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModel

# 防止图片截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 用户配置区域
# ==========================================
DEFAULT_QWEN_PATH = os.environ.get("SILKROAD_VL_CAPTION_MODEL","Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_SIGLIP_PATH = os.environ.get("SILKROAD_SIGLIP_MODEL","google/siglip-so400m-patch14-384")
DEFAULT_INPUT_DIR = os.environ.get("SILKROAD_IMAGES_DIR","data/images")
DEFAULT_FEATURE_PATH = os.environ.get("SILKROAD_SIGLIP_FEATS","outputs/features/siglip_features.pt")" # 需确保此文件已由上一步脚本生成

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
    只加载 SigLIP 用于文本编码，节省显存
    """
    print(f"[{device}] 正在加载 SigLIP (Text Encoder)...", file=sys.stderr)
    try:
        model = AutoModel.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor
    except Exception as e:
        print(f"SigLIP 加载失败: {e}", file=sys.stderr)
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
def rank_with_cache(img_id, img_features_db, candidates, model, processor, device):
    """
    使用预提取的图像特征 + SigLIP 文本编码器进行极速排序
    """
    if not candidates: return "", 0.0, []
    
    # 1. 获取预存的图像特征
    # 逻辑：尝试多种 Key 匹配 (完整ID -> 无后缀文件名 -> 纯文件名)
    img_feat = None
    keys_to_try = [
        img_id, 
        os.path.splitext(img_id)[0], 
        os.path.basename(img_id),
        os.path.splitext(os.path.basename(img_id))[0]
    ]
    
    for k in keys_to_try:
        if k in img_features_db:
            img_feat = img_features_db[k]
            break
            
    if img_feat is None:
        # 如果找不到特征，无法打分，返回第一个候选
        return candidates[0], 0.0, []

    # 转 GPU 并确保归一化
    img_feat = img_feat.to(device)
    img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

    # 2. 编码文本
    try:
        text_inputs = processor(
            text=candidates, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        
        # 3. 矩阵运算计算 Logits
        # SigLIP Logit = (Image . Text) * Scale + Bias
        logit_scale = model.logit_scale.exp()
        logit_bias = getattr(model, 'logit_bias', 0)
        
        # [1, dim] @ [dim, N] -> [1, N]
        logits = (torch.matmul(img_feat, text_feats.t()) * logit_scale) + logit_bias
        
        # 4. Sigmoid
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        
        if isinstance(probs, float): probs = [probs]
        
        # 排序
        scored = list(zip(candidates, probs))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        best_text, best_score = scored[0]
        details = [{"text": t, "siglip_score": round(s, 4)} for t, s in scored]
        
        return best_text, best_score, details

    except Exception as e:
        print(f"Ranking Error: {e}", file=sys.stderr)
        return candidates[0], 0.0, []

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate & Rank (Cached)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--feature_file", type=str, default=DEFAULT_FEATURE_PATH, help="上一步生成的.pt文件")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    parser.add_argument("--qwen_path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--siglip_path", type=str, default=DEFAULT_SIGLIP_PATH)
    parser.add_argument("--image_root", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--num_candidates", type=int, default=5, help="Best-of-N数量")

    args = parser.parse_args()
    set_seed(42 + args.gpu_id)
    
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu_id)

    # 1. 加载预计算特征 (CPU RAM)
    print(f"正在加载图像特征库: {args.feature_file} ...", file=sys.stderr)
    try:
        # map_location='cpu' 避免占用显存
        image_features_db = torch.load(args.feature_file, map_location="cpu")
        print(f"成功加载 {len(image_features_db)} 个图像特征。", file=sys.stderr)
    except Exception as e:
        print(f"特征文件加载失败: {e}", file=sys.stderr)
        print("请先运行 00_extract_image_features.py 生成特征文件！", file=sys.stderr)
        return

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

    # 4. 加载模型 (✅ 已修正：使用 AutoModelForVision2Seq)
    print(f"[{args.gpu_id}] 加载 Qwen...", file=sys.stderr)
    try:
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            args.qwen_path, 
            torch_dtype="auto", 
            device_map=device_str, 
            trust_remote_code=True
        ).eval()
        
        qwen_processor = AutoProcessor.from_pretrained(
            args.qwen_path, 
            min_pixels=256*28*28, 
            max_pixels=1024*28*28, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Qwen Load Error: {e}", file=sys.stderr)
        print("提示: 如果遇到 mrope 错误，请安装 pip install git+https://github.com/huggingface/transformers", file=sys.stderr)
        return

    print(f"[{args.gpu_id}] 加载 SigLIP (Text Part)...", file=sys.stderr)
    siglip_model, siglip_processor = load_siglip_text_only(args.siglip_path, device_str)

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
                continue

            try:
                # 排序 (使用预提取特征)
                # 使用 img_id 或 文件名去匹配特征库
                best_short, s_score, s_details = rank_with_cache(
                    img_id, image_features_db, short_cands, siglip_model, siglip_processor, device_str
                )
                best_long, l_score, l_details = rank_with_cache(
                    img_id, image_features_db, long_cands, siglip_model, siglip_processor, device_str
                )

                result = {
                    "image_id": img_id,
                    "path": full_img_path,
                    "wnid": item.get('wnid', ''),
                    "label_name": item.get('label_name', ''),
                    "width": width,
                    "height": height,
                    "short_caption_best": best_short,
                    "short_score": s_score,
                    "short_candidates": s_details,
                    "long_caption_best": best_long,
                    "long_score": l_score,
                    "long_candidates": l_details
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"Rank/Save Error {img_id}: {e}", file=sys.stderr)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
