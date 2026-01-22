import json
import re
import argparse
import sys
from tqdm import tqdm

# ==========================================
# 配置区域
# ==========================================
TARGET_LANG_KEY = "uzbek"
CYRILLIC_PATTERN = re.compile(r'[\u0400-\u04FF]')

# 定义可能的分数前缀 (根据之前的打分代码)
SCORE_PREFIXES = ["score_bert_", "score_comet_", "score_visual_", "score_kiwi_"]

def load_data(file_path):
    print(f"📖 读取文件: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                return json.loads(content)
            else:
                return [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        sys.exit(1)

def save_data(data, path):
    print(f"💾 保存清洗后的文件至: {path}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def has_cyrillic(text):
    """检测文本是否包含西里尔字母"""
    if not isinstance(text, str) or not text:
        return False
    return bool(CYRILLIC_PATTERN.search(text))

def filter_uzbek(data):
    """
    遍历数据，过滤乌兹别克语中的西里尔文，并同步删除对应的分数。
    """
    print(f"🔍 开始过滤 {TARGET_LANG_KEY} 中的西里尔文及关联分数...")
    
    filtered_count = 0
    total_uzbek_entries = 0
    
    for item in tqdm(data, desc="Processing"):
        if 'translations' not in item:
            continue
        
        translations = item['translations']
        
        if TARGET_LANG_KEY in translations:
            uzbek_data = translations[TARGET_LANG_KEY]
            
            # 使用 list(keys) 创建副本，因为我们可能会在循环中删除分数键
            keys_to_check = list(uzbek_data.keys())
            
            for model_key in keys_to_check:
                val = uzbek_data.get(model_key) # 使用 get 防止键已被删除
                
                # 1. 跳过非字符串 (分数本身在第一轮会被跳过，后面会被主动删除)
                if not isinstance(val, str):
                    continue
                
                # 2. 跳过空字符串
                if not val:
                    continue
                
                # 3. 如果是翻译文本字段 (排除掉可能误判的字符串类型的元数据，虽然一般没有)
                # 简单的办法是：如果字段名本身就是 'score_' 开头，跳过
                if model_key.startswith("score_"):
                    continue

                total_uzbek_entries += 1
                
                # 4. 检测西里尔文
                if has_cyrillic(val):
                    # === 动作 A: 清空文本 ===
                    uzbek_data[model_key] = "" 
                    
                    # === 动作 B: 删除关联分数 ===
                    # 逻辑：如果文本键是 "short_nllb"，分数键通常是 "score_bert_short_nllb"
                    for prefix in SCORE_PREFIXES:
                        score_key = f"{prefix}{model_key}"
                        
                        # 如果存在这个分数键，将其删除 (或者置为 -1)
                        if score_key in uzbek_data:
                            # 方式1: 直接删除键 (推荐，保持数据干净)
                            del uzbek_data[score_key]
                            
                            # 方式2: 置为 -1 (如果你希望保留键)
                            # uzbek_data[score_key] = -1.0
                            
                    filtered_count += 1

    return data, filtered_count, total_uzbek_entries

def main():
    parser = argparse.ArgumentParser(description="过滤乌兹别克语西里尔文并清除分数")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()

    data = load_data(args.input_file)
    cleaned_data, filtered_num, total_num = filter_uzbek(data)
    
    print("\n" + "="*40)
    print(f"📊 过滤统计报告")
    print(f"="*40)
    print(f"🔹 处理总条目数 (Rows): {len(data)}")
    print(f"🔹 检查的翻译字段数: {total_num}")
    print(f"🔻 清洗掉的条目数 (含分数): {filtered_num}")
    if total_num > 0:
        print(f"📉 过滤比例: {(filtered_num / total_num) * 100:.2f}%")
    print("="*40 + "\n")

    save_data(cleaned_data, args.output_file)
    print("🎉 完成！数据已清洗，对应分数已移除。")

if __name__ == "__main__":
    main()
