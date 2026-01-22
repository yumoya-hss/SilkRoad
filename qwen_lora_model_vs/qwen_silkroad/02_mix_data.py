import json
import os
import random

# ================= 🔴 配置区域 🔴 =================
# 1. Split 后的根目录 (里面应该有 train, val, test 三个文件夹)
SPLIT_ROOT = "/home/houshuoshuo/qlora_data/split"

# 2. 图片根目录 (ImageNet50K)
IMAGE_ROOT = "/mnt/raid/hss/dataset/Image50K"

# 3. 输出文件保存目录 (建议就保存在 split 根目录下，方便管理)
OUTPUT_DIR = "/home/houshuoshuo/qlora_data/split"

# 随机种子 (保证打乱顺序一致)
SEED = 42
# ===================================================

def convert_single_split(split_name):
    """
    处理单个划分 (如 'train', 'val', 'test')
    """
    input_dir = os.path.join(SPLIT_ROOT, split_name)
    output_file = os.path.join(OUTPUT_DIR, f"silkroad_{split_name}.json")
    
    all_data = []

    # 语言代码映射表
    lang_map = {
        'ug': 'Uyghur', 'uyghur': 'Uyghur',
        'uz': 'Uzbek', 'uzbek': 'Uzbek',
        'kk': 'Kazakh', 'kazakh': 'Kazakh',
        'ur': 'Urdu', 'urdu': 'Urdu',
        'ky': 'Kyrgyz', 'kyrgyz': 'Kyrgyz',
        'tg': 'Tajik', 'tajik': 'Tajik'
    }

    if not os.path.exists(input_dir):
        print(f"❌ Error: 找不到目录 {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    # 排序以保证处理顺序一致
    files.sort()
    
    print(f"🚀 正在处理 [{split_name}] 集: 扫描到 {len(files)} 个文件...")

    for filename in files:
        # 修正文件名解析逻辑：
        # 上一步切分生成的可能是 "kazakh.json"，没有下划线了
        # 所以直接去掉 .json 后缀即可拿到语言名
        lang_key = filename.replace('.json', '').lower()
        
        # 兼容逻辑：万一文件名里还有下划线 (如 dataset_kazakh.json)，尝试提取
        if '_' in lang_key:
             # 这里假设语言名在最后，或者你自己根据实际情况调整
             # 比如 dataset_kazakh -> kazakh
             parts = lang_key.split('_')
             # 简单的启发式：看哪部分在映射表里
             found = False
             for p in parts:
                 if p in lang_map:
                     lang_key = p
                     found = True
                     break
             if not found:
                 lang_key = parts[0] # 默认取第一部分

        target_lang = lang_map.get(lang_key, "Target Language")

        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # print(f"   -> 读取 {filename} ({target_lang}): {len(data)} 条")

        for item in data:
            # 构造绝对图片路径
            img_filename = os.path.basename(item.get('path', ''))
            abs_img_path = os.path.join(IMAGE_ROOT, img_filename)

            source_text = item.get('src_text', '')
            target_text = item.get('tgt_text', '')

            if not source_text or not target_text: continue

            # 构造 Prompt
            conversation = [
                {
                    "from": "human",
                    "value": f"<image>\nPlease translate the description of this image into {target_lang}.\nEnglish Source: {source_text}"
                },
                {
                    "from": "gpt",
                    "value": target_text
                }
            ]

            all_data.append({
                "images": [abs_img_path],
                "conversations": conversation
            })

    # 打乱数据
    random.seed(SEED)
    random.shuffle(all_data)

    print(f"✅ [{split_name}] 合并完成！总共 {len(all_data)} 条样本。")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存: {output_file}\n")

if __name__ == "__main__":
    # 批量处理三个文件夹
    for split in ['train', 'val', 'test']:
        convert_single_split(split)
