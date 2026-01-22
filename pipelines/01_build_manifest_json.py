import json
import os
import argparse
from PIL import Image
from tqdm import tqdm

def load_mappings(wnid_txt_path, class_index_json_path):
    """
    加载映射关系
    1. 从 json 加载 WNID -> Label Name 的字典
    2. 从 txt 加载验证集顺序的 WNID 列表
    """
    print(f"Loading mappings from {class_index_json_path}...")
    with open(class_index_json_path, 'r', encoding='utf-8') as f:
        raw_index = json.load(f)

    # 构建查找表: {'n01751748': 'sea_snake', ...}
    wnid_to_name = {}
    for idx_str, (wnid, name) in raw_index.items():
        wnid_to_name[wnid] = name.replace('_', ' ') # 将下划线替换为空格，更易读

    print(f"Loading validation labels from {wnid_txt_path}...")
    with open(wnid_txt_path, 'r', encoding='utf-8') as f:
        # 读取每一行并去除空白符
        val_wnids = [line.strip() for line in f.readlines() if line.strip()]

    if len(val_wnids) != 50000:
        print(f"Warning: Expected 50,000 labels, but found {len(val_wnids)} in txt file.")

    return wnid_to_name, val_wnids

def generate_manifest(args):
    # 1. 加载映射
    wnid_to_name, val_wnids = load_mappings(args.labels_txt, args.class_index)

    manifest_data = []

    print("Processing images...")
    # 遍历 50000 个 WNID，索引 i 从 0 开始，所以图片 ID 是 i+1
    for i, wnid in tqdm(enumerate(val_wnids), total=len(val_wnids)):
        # 构造 ImageNet 验证集的标准文件名: ILSVRC2012_val_00000001.JPEG
        idx = i + 1
        image_id = f"ILSVRC2012_val_{idx:08d}"
        filename = f"{image_id}.JPEG"

        # 拼接相对路径 (用于写入 manifest) 和 绝对路径 (用于读取宽高)
        relative_path = os.path.join(args.image_folder_name, filename).replace("\\", "/") # 保证路径兼容
        abs_path = os.path.join(args.dataset_root, filename)

        # 获取 Label Name
        label_name = wnid_to_name.get(wnid, "Unknown")

        # 获取图片信息
        width, height = 0, 0
        bad_image = False

        # 尝试读取图片宽高
        if os.path.exists(abs_path):
            try:
                # 懒加载打开，不完全解码，速度较快
                with Image.open(abs_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                bad_image = True
        else:
            # 如果本地没图片（比如你只跑脚本没下图），为了生成 manifest 可以容忍，但标记一下
            # print(f"File not found: {abs_path}")
            bad_image = True

        # 构造单条记录
        record = {
            "image_id": image_id,
            "path": relative_path,  # e.g., "dataset/ILSVRC2012_val_00000001.JPEG"
            "wnid": wnid,
            "label_name": label_name,
            "width": width,
            "height": height,
            # "bad_image": bad_image
        }

        manifest_data.append(record)

    # 写入 JSON 文件
    output_path = args.output_path
    print(f"Writing {len(manifest_data)} records to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 输入文件路径配置
    parser.add_argument("--labels_txt", type=str,
                        default="data/metadata/imagenet_2012_validation_synset_labels.txt",
                        help="Path to the txt file containing ordered WNIDs")

    parser.add_argument("--class_index", type=str,
                        default="data/metadata/imagenet_class_index.json",
                        help="Path to the class index json file")

    # 图片目录配置
    parser.add_argument("--dataset_root", type=str,
                        default="data/metadata/ILSVRC2012_img_val",
                        help="Local folder where images are actually stored")

    # 输出 JSON 中的 path 字段前缀
    parser.add_argument("--image_folder_name", type=str,
                        default="dataset",
                        help="The folder name prefix to write in the JSON 'path' field")
    parser.add_argument("--output_path", type=str, default=os.environ.get("SILKROAD_MANIFEST","outputs/manifest/manifest.json"))

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.labels_txt) or not os.path.exists(args.class_index):
        print("Error: Please ensure input txt and json files exist.")
    else:
        generate_manifest(args)