import torch
import os
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default=os.environ.get("SILKROAD_IMAGES_DIR","data/images"))
    parser.add_argument("--save_path", type=str, default=os.environ.get("SILKROAD_SIGLIP_FEATS","outputs/features/siglip_features.pt"))
    parser.add_argument("--siglip_path", type=str, default=os.environ.get("SILKROAD_SIGLIP_MODEL","google/siglip-so400m-patch14-384"))
    parser.add_argument("--batch_size", type=int, default=64) # SigLIP 可以开很大 batch
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型 (只需要 Vision 部分，但 AutoModel 加载全部也无妨)
    print("Loading SigLIP...")
    model = AutoModel.from_pretrained(args.siglip_path, device_map=device, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(args.siglip_path, trust_remote_code=True)

    # 2. 获取图片列表
    images = sorted([f for f in os.listdir(args.image_root) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # 3. 批量提取
    feature_dict = {}
    
    print(f"Start extracting features for {len(images)} images...")
    # 简单的 Batch 处理循环
    for i in tqdm(range(0, len(images), args.batch_size)):
        batch_files = images[i : i + args.batch_size]
        batch_paths = [os.path.join(args.image_root, f) for f in batch_files]
        
        # 批量读图
        try:
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        except:
            print(f"Error reading batch {i}, skipping...")
            continue

        # 预处理
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # 获取图像特征
            image_features = model.get_image_features(**inputs)
            # 归一化 (这一步很重要，存下来归一化后的特征，后续直接点积)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 转 CPU 存入字典以节省显存
            image_features = image_features.cpu()
            
        for fname, feat in zip(batch_files, image_features):
            # Key 可以是文件名或去除后缀的 ID
            key = os.path.splitext(fname)[0] 
            feature_dict[key] = feat

    # 4. 保存
    print(f"Saving {len(feature_dict)} features to {args.save_path}...")
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(feature_dict, args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()