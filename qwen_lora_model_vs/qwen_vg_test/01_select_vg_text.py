import json
import os
import random
from tqdm import tqdm
from PIL import Image  # ✅ 引入图像处理库

# ================= ⚙️ 配置区域 =================
# 1. 原始 JSONL 文件路径
SOURCE_JSONL = "/mnt/raid/wjx/vision-rag-mt/kb/bn/bn_vg_test.jsonl"

# 2. 原始图片根目录
SOURCE_IMAGE_DIR = "/mnt/raid/wjx/vision-rag-mt/images"

# 3. 输出保存的根目录
OUTPUT_ROOT = "/home/houshuoshuo/qlora_data/test/vg_50_crop_dataset"

# 4. 目标抽取数量
TARGET_COUNT = 100
# ===============================================

def process_dataset():
    # 1. 准备输出目录
    output_img_dir = os.path.join(OUTPUT_ROOT, "images")
    output_json_path = os.path.join(OUTPUT_ROOT, "vg_50_crop.json")
    
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_img_dir}")

    # 2. 读取原始 JSONL 数据
    print(f"📖 正在读取: {SOURCE_JSONL}")
    if not os.path.exists(SOURCE_JSONL):
        print(f"❌ 错误: 找不到源文件 {SOURCE_JSONL}")
        return

    with open(SOURCE_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. 随机打乱
    print(f"🎲 数据总数: {len(lines)}，正在随机打乱...")
    random.shuffle(lines)

    collected_data = []
    count = 0

    # 4. 遍历并处理
    print("🚀 开始提取并裁剪图片...")
    for line in tqdm(lines):
        if count >= TARGET_COUNT:
            break
            
        try:
            item = json.loads(line)
            
            # 提取字段
            image_id = item.get('image_id')
            src_en = item.get('src_en')
            bbox = item.get('bbox')
            sample_index = item.get('sample_index', count) #以此作为唯一标识防止覆盖
            
            if not image_id or not src_en or not bbox:
                continue

            # 源图片路径
            src_img_filename = f"{image_id}.jpg"
            src_img_path = os.path.join(SOURCE_IMAGE_DIR, src_img_filename)

            if os.path.exists(src_img_path):
                # ✅ 打开原始图片
                with Image.open(src_img_path) as img:
                    img = img.convert("RGB") # 确保兼容性
                    
                    # ✅ 计算裁剪坐标
                    # bbox格式通常是: x(左上角横坐标), y(左上角纵坐标), w(宽), h(高)
                    x = bbox['x']
                    y = bbox['y']
                    w = bbox['w']
                    h = bbox['h']
                    
                    # PIL crop 需要 (left, top, right, bottom)
                    left = x
                    top = y
                    right = x + w
                    bottom = y + h
                    
                    # 执行裁剪
                    cropped_img = img.crop((left, top, right, bottom))
                    
                    # ✅ 生成新的文件名 
                    # 注意：因为一张图可能有多个框，必须加上 sample_index 区分，否则会互相覆盖
                    new_filename = f"{image_id}_{sample_index}.jpg"
                    dst_img_path = os.path.join(output_img_dir, new_filename)
                    
                    # 保存裁剪后的图片
                    cropped_img.save(dst_img_path)
                
                # 构建目标 JSON 格式
                entry = {
                    "image_filename": new_filename, # 使用新的文件名
                    "origin_image_id": image_id,    # 保留原始ID备查
                    "image_id": f"{image_id}_{sample_index}", # 生成新的唯一ID
                    "src_en": src_en,
                    "saved_path": os.path.abspath(dst_img_path)
                }
                
                collected_data.append(entry)
                count += 1

        except Exception as e:
            # print(f"⚠️ 跳过错误数据: {e}") 
            continue

    # 5. 保存 JSON
    print(f"\n💾 正在保存 JSON 到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成！共提取并裁剪 {len(collected_data)} 条数据。")
    print(f"   裁剪图片位置: {output_img_dir}")
    print(f"   JSON位置: {output_json_path}")

if __name__ == "__main__":
    process_dataset()
