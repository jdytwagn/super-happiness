import os
import shutil
import random


def create_yolov8_structure(dataset_file, base_dir):
    # 定义 YOLOv8 目录结构
    images_train_dir = os.path.join(base_dir, 'images', 'train')
    images_val_dir = os.path.join(base_dir, 'images', 'val')
    labels_train_dir = os.path.join(base_dir, 'labels', 'train')
    labels_val_dir = os.path.join(base_dir, 'labels', 'val')

    # 创建目录
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # 读取数据集文本文档
    with open(dataset_file, 'r') as f:
        lines = f.readlines()

    # 打乱数据集
    random.shuffle(lines)

    # 按行处理
    for index, line in enumerate(lines):
        img_path, category = line.strip().split()
        category = int(category)

        # 假设标签文件名与图像文件名相同，但扩展名为 .txt
        label_filename = os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt')

        # 80% 的数据用于训练，20% 的数据用于验证
        if index < len(lines) * 0.8:
            dest_img_dir = images_train_dir
            dest_label_dir = labels_train_dir
        else:
            dest_img_dir = images_val_dir
            dest_label_dir = labels_val_dir

        # 复制图像文件
        shutil.copy(img_path, dest_img_dir)

        # 生成标签文件
        label_path = os.path.join(dest_label_dir, label_filename)
        with open(label_path, 'w') as label_file:
            # 整个图像是一个类别，中心点在 (0.5, 0.5)，宽度和高度为 1
            label_file.write(f"{category} 0.5 0.5 1 1\n")


# 调用函数
dataset_file = 'dataset.txt'  # 替换为你的数据集文本文档路径
base_dir = 'yolov8_dataset'  # YOLOv8 需要的目录结构的根目录
create_yolov8_structure(dataset_file, base_dir)
