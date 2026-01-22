import json
import os
import sys
from tqdm import tqdm

# ==========================================
# 🛠️ 配置区域：请在这里填入那 6 个文件的路径
# ==========================================
INPUT_FILES = {
    "uyghur":  "./qwen_translation_uyghur.json",
    "uzbek":   "./qwen_translation_uzbek.json",
    "kazakh":  "./qwen_translation_kazakh.json",
    "kyrgyz":  "./qwen_translation_kyrgyz.json",
    "tajik":   "./qwen_translation_tajik.json",
    "urdu":    "./qwen_translation_urdu.json"
}

# 输出文件路径
OUTPUT_FILE = "./qwen_translation.json"

# ==========================================
# 工具函数
# ==========================================
def load_data(file_path):
    print(f"📖 Loading: {file_path} ...")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content: return []
            if content.startswith('['):
                return json.loads(content)
            else:
                return [json.loads(line) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"❌ JSON 解析错误: {e}")
        sys.exit(1)

def main():
    # 1. 选取第一个文件作为“基准” (Base)
    # 我们将把其他文件的内容合并到这个基准数据中
    base_lang = list(INPUT_FILES.keys())[0]
    base_path = INPUT_FILES[base_lang]
    
    print(f"🏗️ 初始化基准数据，使用语言: {base_lang}")
    merged_data = load_data(base_path)
    
    # 建立索引映射： image_id -> list_index
    # 这样可以实现 O(1) 的快速查找，不用双重循环，防止数据量大时卡死
    id_map = {}
    for idx, item in enumerate(merged_data):
        # 假设每个 item 都有唯一的 'image_id' 或 'id'
        # 如果您的数据没有 image_id，请确保所有文件顺序完全一致，则不需要 id_map，直接按 index 合并
        key = item.get('image_id', item.get('id', str(idx))) 
        id_map[str(key)] = idx

    # 2. 遍历剩余的 5 个文件进行合并
    for lang, file_path in INPUT_FILES.items():
        if lang == base_lang:
            continue # 跳过基准语言
            
        print(f"🔄 正在合并语言: {lang} ...")
        current_data = load_data(file_path)
        
        # 遍历当前语言的数据
        match_count = 0
        for item in tqdm(current_data, desc=f"Merging {lang}"):
            # 找到对应的 key
            key = str(item.get('image_id', item.get('id', "")))
            
            # 如果没有 ID，尝试用顺序匹配（仅当您确定顺序绝对一致时）
            # 这里默认使用 ID 匹配更安全
            
            if key in id_map:
                target_idx = id_map[key]
                target_item = merged_data[target_idx]
                
                # 确保目标有 translations 字段
                if 'translations' not in target_item:
                    target_item['translations'] = {}
                
                # 提取当前文件中的该语言翻译
                # 结构通常是 item['translations'][lang] = {...}
                if 'translations' in item and lang in item['translations']:
                    target_item['translations'][lang] = item['translations'][lang]
                    match_count += 1
            else:
                # 如果找不到 ID，说明数据不齐
                pass
        
        print(f"   ✅ 成功合并 {match_count} 条 {lang} 数据")

    # 3. 保存最终结果
    print(f"\n💾 正在保存最终文件至: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print("🎉 合并完成！所有 6 种语言已整合到一个 JSON 中。")

if __name__ == "__main__":
    main()