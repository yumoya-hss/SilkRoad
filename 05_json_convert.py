import json
import re
from tqdm import tqdm

def load_multi_line_json(input_file):
    """
    读取多行完整JSON字典（解决换行拆分问题）
    """
    # 读取全部内容并合并换行
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n', '').replace('\r', '').strip()
    
    # 关键修复：匹配最外层的大字典（排除嵌套的小字典）
    # 正则逻辑：匹配 { 开头，直到 最后一个 } 结束的完整外层字典
    # 处理多个外层字典拼接的情况（无逗号分隔）
    pattern = r'\{(?:[^{}]|(\{(?:[^{}]|(\{[^{}]*\})*)*\}))*\}'
    matches = re.findall(pattern, content)
    
    # 提取真正的匹配结果（正则分组问题，取原始匹配）
    # 重新遍历，准确捕获每个外层字典
    outer_matches = []
    idx = 0
    while idx < len(content):
        # 找下一个 { 的位置
        start = content.find('{', idx)
        if start == -1:
            break
        # 匹配对应的 }（处理嵌套）
        end = start
        bracket_count = 1
        while bracket_count > 0 and end < len(content)-1:
            end += 1
            if content[end] == '{':
                bracket_count += 1
            elif content[end] == '}':
                bracket_count -= 1
        # 提取完整字典
        outer_dict = content[start:end+1]
        outer_matches.append(outer_dict)
        idx = end + 1
    
    return outer_matches

def convert_to_target_format(input_file, output_file):
    """
    完整处理流程：读取→解析→转换→保存（带进度条）
    """
    # 1. 读取文件并提取外层字典
    try:
        print("📌 开始读取并提取外层JSON字典...")
        outer_matches = load_multi_line_json(input_file)
        print(f"✅ 共提取到 {len(outer_matches)} 个外层字典")
    except FileNotFoundError:
        print(f"❌ 错误：文件 {input_file} 不存在！")
        return
    except Exception as e:
        print(f"❌ 读取文件失败：{e}")
        return
    
    # 2. 解析每个外层字典（带进度条）
    data_list = []
    for idx, raw_dict in enumerate(tqdm(outer_matches, desc="解析完整字典", unit="条")):
        try:
            # 修复格式问题（单引号→双引号、多余逗号）
            fixed_dict = raw_dict.replace("'", '"')
            fixed_dict = re.sub(r',\s*}', '}', fixed_dict)  # 移除末尾多余逗号
            # 解析为JSON字典
            item = json.loads(fixed_dict)
            data_list.append(item)
        except json.JSONDecodeError as e:
            tqdm.write(f"⚠️  第{idx+1}条解析失败：{e}")
            tqdm.write(f"   预览：{raw_dict[:300]}...")
            continue
    
    if not data_list:
        print("❌ 无有效解析数据！")
        return
    print(f"✅ 成功解析 {len(data_list)} 条有效数据")
    
    # 3. 转换为目标格式（带进度条）
    converted = []
    for item in tqdm(data_list, desc="转换数据格式", unit="条"):
        converted.append({
            "image_id": item.get("image_id", ""),
            "path": item.get("path", ""),
            "wnid": item.get("wnid", ""),
            "label_name": item.get("label_name", ""),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
            "short_caption_best": item.get("short_caption_best", ""),
            "short_score": item.get("short_score", 0.0),
            "short_candidates": item.get("short_candidates", []),
            "long_caption_best": item.get("long_caption_best", ""),
            "long_score": item.get("long_score", 0.0),
            "long_candidates": item.get("long_candidates", [])
        })
    
    # 4. 保存为标准JSON数组（[{}, {}]）
    try:
        json_str = json.dumps(converted, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        total_size = len(json_bytes)
        
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, desc="保存文件", unit="B", unit_scale=True) as pbar:
                chunk_size = 4096
                for i in range(0, total_size, chunk_size):
                    chunk = json_bytes[i:i+chunk_size]
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"✅ 转换完成！文件保存至：{output_file}")
    except Exception as e:
        print(f"❌ 保存失败：{e}")

# 主函数
if __name__ == "__main__":
    # 自动安装tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("📦 安装tqdm...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
    
    # 你的文件路径
    INPUT_JSON = "manifest_best.json"
    OUTPUT_JSON = "Image_En_data.json"
    
    convert_to_target_format(INPUT_JSON, OUTPUT_JSON)
