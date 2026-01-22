import os
import json
import random
from pathlib import Path

# ====================== 配置区 ======================
# JSON 数据目录（绝对路径）
JSON_DIR = "outputs/fr"
# 原始图片根目录
IMAGE_ROOT_DIR = "outputs/images_multi30k"
# 随机选取的图片保存目录（自动创建）
SELECTED_IMAGE_DIR = "/outputs/multi_random_100"
# 输出结果文件
OUTPUT_FILE = "multi_100_en.json"
# 随机种子（保证每次选取结果一致）
RANDOM_SEED = 42
# 要选取的图片数量
SELECT_IMAGE_NUM = 100
# ====================================================

def get_image_id_from_filename(filename):
    """从图片文件名提取image_id（处理各种后缀）"""
    try:
        # 去掉所有后缀，提取纯数字image_id
        img_id_str = Path(filename).stem
        return int(img_id_str)
    except ValueError:
        return None

def main():
    # 1. 设置随机种子，保证结果可复现
    random.seed(RANDOM_SEED)
    print(f"🔒 随机种子已设置为: {RANDOM_SEED}")

    # 2. 读取并合并所有 JSON Lines 文件（.jsonl）
    merged_data = []
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".jsonl")]
    if not json_files:
        print(f"❌ 在 {JSON_DIR} 未找到任何 .jsonl 文件")
        return
    
    for filename in json_files:
        json_path = os.path.join(JSON_DIR, filename)
        print(f"📖 正在读取: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    merged_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"⚠️  跳过 {json_path} 第 {line_num} 行无效JSON: {e}")
    
    if not merged_data:
        print("❌ 未读取到任何有效JSON数据")
        return
    print(f"✅ 已合并 {len(merged_data)} 条JSON数据")

    # 3. 建立 image_id -> src_en 的映射，同时收集所有有效image_id
    image_id_to_src_en = {}
    valid_image_ids = set()
    for item in merged_data:
        image_id = item.get("image_id")
        src_en = item.get("src_en")
        if image_id and src_en:
            image_id_to_src_en[image_id] = src_en
            valid_image_ids.add(image_id)
    
    print(f"✅ 建立了 {len(image_id_to_src_en)} 个image_id与英文描述的映射")

    # 4. 扫描图片目录，筛选出有对应描述的图片
    all_images = []
    for img_filename in os.listdir(IMAGE_ROOT_DIR):
        img_path = os.path.join(IMAGE_ROOT_DIR, img_filename)
        # 只处理文件，跳过目录
        if not os.path.isfile(img_path):
            continue
        
        # 提取image_id并检查是否有对应描述
        image_id = get_image_id_from_filename(img_filename)
        if image_id and image_id in valid_image_ids:
            all_images.append({
                "filename": img_filename,
                "path": img_path,
                "image_id": image_id
            })
    
    if len(all_images) < SELECT_IMAGE_NUM:
        print(f"⚠️  有对应描述的图片仅 {len(all_images)} 张，不足 {SELECT_IMAGE_NUM} 张，将选取全部")
        selected_images = all_images
    else:
        # 随机选取指定数量的图片
        selected_images = random.sample(all_images, SELECT_IMAGE_NUM)
    
    print(f"✅ 随机选取了 {len(selected_images)} 张有对应描述的图片")

    # 5. 创建选中图片的保存目录
    os.makedirs(SELECTED_IMAGE_DIR, exist_ok=True)
    
    # 6. 复制选中的图片到目标目录，并收集匹配结果
    matched_results = []
    for img_info in selected_images:
        img_filename = img_info["filename"]
        img_path = img_info["path"]
        image_id = img_info["image_id"]
        
        # 复制图片到目标目录
        target_img_path = os.path.join(SELECTED_IMAGE_DIR, img_filename)
        try:
            import shutil
            shutil.copy2(img_path, target_img_path)  # 保留文件元数据
        except Exception as e:
            print(f"❌ 复制图片失败 {img_filename}: {e}")
            continue
        
        # 获取对应的英文描述
        src_en = image_id_to_src_en[image_id]
        matched_results.append({
            "image_filename": img_filename,
            "image_id": image_id,
            "src_en": src_en,
            "saved_path": target_img_path
        })
        print(f"✅ 处理完成: {img_filename} -> {src_en[:60]}...")

    # 7. 保存匹配结果到文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(matched_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 任务全部完成！")
    print(f"📁 选中的图片已保存到: {SELECTED_IMAGE_DIR}")
    print(f"📄 匹配结果已保存到: {OUTPUT_FILE}")
    print(f"🔍 共成功处理 {len(matched_results)} 张图片（均有对应英文描述）")

if __name__ == "__main__":
    main()
