import json
import os

def load_json(file_path):
    """加载JSON文件"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_lookup_dict(data_list):
    """将列表转换为以 image_id 为键的字典，方便快速查找"""
    return {item['image_id']: item for item in data_list}

def merge_datasets():
    # 1. 定义文件名
    main_file = 'translated_data.json'
    madlad_file = 'madlad_translation.json'
    qwen_file = 'qwen_translation.json'
    output_file = 'translated_data_updated.json' #为了安全，建议先存为新文件

    # 2. 加载数据
    print("正在加载数据...")
    main_data = load_json(main_file)
    madlad_data = load_json(madlad_file)
    qwen_data = load_json(qwen_file)

    # 3. 创建查找表 (Hash Map) 以提高性能
    print("正在构建索引...")
    madlad_lookup = create_lookup_dict(madlad_data)
    qwen_lookup = create_lookup_dict(qwen_data)

    # 4. 开始合并
    print("正在合并数据...")
    count_updated = 0
    
    for entry in main_data:
        img_id = entry.get('image_id')
        if not img_id:
            continue

        # 获取当前条目的翻译字典
        # 如果原始数据中没有translations字段，初始化为空字典
        current_translations = entry.get('translations', {})

        # --- 合并 Madlad 数据 ---
        if img_id in madlad_lookup:
            madlad_entry = madlad_lookup[img_id]
            madlad_trans = madlad_entry.get('translations', {})
            
            # 遍历Madlad中的每种语言 (uyghur, uzbek...)
            for lang, trans_content in madlad_trans.items():
                if lang not in current_translations:
                    current_translations[lang] = {}
                # 更新该语言下的字段 (例如添加 short_madlad, long_madlad)
                current_translations[lang].update(trans_content)

        # --- 合并 Qwen 数据 ---
        if img_id in qwen_lookup:
            qwen_entry = qwen_lookup[img_id]
            qwen_trans = qwen_entry.get('translations', {})
            
            # 遍历Qwen中的每种语言
            for lang, trans_content in qwen_trans.items():
                if lang not in current_translations:
                    current_translations[lang] = {}
                # 更新该语言下的字段 (例如添加 short_qwen, long_qwen)
                current_translations[lang].update(trans_content)
        
        count_updated += 1

    # 5. 保存结果
    print(f"合并完成。共处理 {count_updated} 条数据。")
    print(f"正在写入文件: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)
    
    print("完成！")

if __name__ == '__main__':
    merge_datasets()