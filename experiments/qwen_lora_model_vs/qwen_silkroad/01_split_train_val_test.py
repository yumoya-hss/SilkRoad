import os
import json
import random
import shutil

# ================= 🔴 配置 =================
SOURCE_DIR = "outputs/final_datasets_split"
OUTPUT_ROOT = "outputs/split"
TEST_SAMPLES_PER_LANG = 250
VAL_SAMPLES_PER_LANG = 50

# 🔒 随机种子
SEED = 42
# ===========================================

def split_dataset():
    # 1. 锁定随机种子
    random.seed(SEED)
    print(f"🔒 随机种子已锁定为: {SEED} (保证每次划分结果一致)")

    for sub in ['train', 'val', 'test']:
        path = os.path.join(OUTPUT_ROOT, sub)
        os.makedirs(path, exist_ok=True)

    # 2. 读取并排序文件
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    files.sort()

    lang_files = {}
    for f in files:
        # 🔴 修改核心：针对 dataset_kazakh_long.json 这种文件名
        # split('_') 得到 ['dataset', 'kazakh', 'long.json']
        # 我们取 [1] 也就是 'kazakh'
        try:
            lang = f.split('_')[1].lower()
        except IndexError:
            # 防止万一有个文件叫 data.json 这种没有下划线的，做个保底
            lang = f.split('.')[0].lower()

        if lang not in lang_files: lang_files[lang] = []
        lang_files[lang].append(f)

    print(f"🌍 识别到 {len(lang_files)} 种语言: {list(lang_files.keys())}")

    # 3. 开始分层处理
    for lang, file_list in lang_files.items():
        all_items = []
        for fname in file_list:
            with open(os.path.join(SOURCE_DIR, fname), 'r', encoding='utf-8') as f:
                all_items.extend(json.load(f))

        grouped_by_img = {}
        for item in all_items:
            img_path = item.get('path')
            if img_path not in grouped_by_img: grouped_by_img[img_path] = []
            grouped_by_img[img_path].append(item)

        # 排序再打乱
        unique_images = list(grouped_by_img.keys())
        unique_images.sort()
        random.shuffle(unique_images)

        total_imgs = len(unique_images)
        print(f"   Processing {lang}: 总图片数 {total_imgs} ...")

        n_test_imgs = TEST_SAMPLES_PER_LANG
        n_val_imgs = VAL_SAMPLES_PER_LANG

        if total_imgs < (n_test_imgs + n_val_imgs) * 2:
            print(f"   ⚠️ 警告: {lang} 数据过少，切换为 10% 测试集模式")
            n_test_imgs = int(total_imgs * 0.1)
            n_val_imgs = int(total_imgs * 0.05)

        test_img_ids = unique_images[:n_test_imgs]
        val_img_ids = unique_images[n_test_imgs: n_test_imgs + n_val_imgs]
        train_img_ids = unique_images[n_test_imgs + n_val_imgs:]

        def flatten_data(img_ids):
            data = []
            for img_id in img_ids:
                data.extend(grouped_by_img[img_id])
            return data

        test_data = flatten_data(test_img_ids)
        val_data = flatten_data(val_img_ids)
        train_data = flatten_data(train_img_ids)

        with open(os.path.join(OUTPUT_ROOT, 'test', f'{lang}.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        with open(os.path.join(OUTPUT_ROOT, 'val', f'{lang}.json'), 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        with open(os.path.join(OUTPUT_ROOT, 'train', f'{lang}.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        print(f"     -> Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    print("\n✅ 所有语言分层切分完成！")

if __name__ == "__main__":
    split_dataset()
