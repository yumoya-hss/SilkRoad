import json
import os

def load_json(file_path):
    """加载JSON文件"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取错误 {file_path}: {e}")
        return []

def create_lookup_dict(data_list):
    """将列表转换为以 image_id 为键的字典，方便快速查找"""
    return {item['image_id']: item for item in data_list}

def merge_bt_datasets():
    # ================= 配置区域 =================
    # 1. 定义文件名 (根据您的要求修改)
    main_file = 'dataset_with_bt.json'               # 主文件 (通常包含 NLLB 回译)
    madlad_file = 'madlad_translation_with_bt.json'  # Madlad 回译文件
    qwen_file = 'qwen_translation_bt.json'           # Qwen 回译文件
    
    output_file = 'translated_data_bt.json'          # 输出文件
    # ===========================================

    # 2. 加载数据
    print(f"正在加载主文件: {main_file} ...")
    main_data = load_json(main_file)
    
    print(f"正在加载 Madlad 文件: {madlad_file} ...")
    madlad_data = load_json(madlad_file)
    
    print(f"正在加载 Qwen 文件: {qwen_file} ...")
    qwen_data = load_json(qwen_file)

    if not main_data:
        print("主文件加载失败或为空，程序终止。")
        return

    # 3. 创建查找表 (Hash Map) 以提高性能
    print("正在构建索引...")
    madlad_lookup = create_lookup_dict(madlad_data)
    qwen_lookup = create_lookup_dict(qwen_data)

    # 4. 开始合并
    print("正在合并回译数据...")
    count_updated = 0
    
    for entry in main_data:
        img_id = entry.get('image_id')
        if not img_id:
            continue

        # 获取当前条目的翻译字典
        # 结构通常是: entry['translations']['uyghur'] = { ... }
        current_translations = entry.get('translations', {})

        # --- 合并 Madlad 回译数据 ---
        if img_id in madlad_lookup:
            madlad_entry = madlad_lookup[img_id]
            madlad_trans = madlad_entry.get('translations', {})
            
            # 遍历Madlad中的每种语言
            for lang, trans_content in madlad_trans.items():
                if lang not in current_translations:
                    current_translations[lang] = {}
                # update 会自动将 bt_short_madlad 等新字段加入到该语言字典中
                current_translations[lang].update(trans_content)

        # --- 合并 Qwen 回译数据 ---
        if img_id in qwen_lookup:
            qwen_entry = qwen_lookup[img_id]
            qwen_trans = qwen_entry.get('translations', {})
            
            # 遍历Qwen中的每种语言
            for lang, trans_content in qwen_trans.items():
                if lang not in current_translations:
                    current_translations[lang] = {}
                # update 会自动将 bt_short_qwen 等新字段加入到该语言字典中
                current_translations[lang].update(trans_content)
        
        count_updated += 1

    # 5. 保存结果
    print(f"合并完成。共处理 {count_updated} 条数据。")
    print(f"正在写入文件: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(main_data, f, ensure_ascii=False, indent=2)
        print("✅ 完成！所有回译数据已合并。")
    except Exception as e:
        print(f"❌ 写入失败: {e}")

if __name__ == '__main__':
    merge_bt_datasets()