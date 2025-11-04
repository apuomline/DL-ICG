import os
import shutil
import csv
import argparse
import yaml
from pathlib import Path

def copy_images_from_csv(config):
    """
    根据YAML配置从源数据集中复制图像到指定位置
    跳过指定文件列表中的图像
    
    Args:
        config: 包含所有配置参数的字典
    """
    # 从配置中获取参数
    csv_file = config.get('csv_file')
    source_data_dir = config.get('source_dir')
    skip_file_path = config.get('skip_file')
    recursive_search = config.get('recursive_search', True)
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件 {csv_file} 不存在")
        return
    
    # 检查源数据集目录是否存在
    if not os.path.exists(source_data_dir):
        print(f"错误: 源数据集目录 {source_data_dir} 不存在")
        return
    
    # 读取需要跳过的文件列表
    skip_files = set()
    if skip_file_path and os.path.exists(skip_file_path):
        with open(skip_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if filename:  # 跳过空行
                    skip_files.add(filename)
        print(f"从 {skip_file_path} 中读取了 {len(skip_files)} 个需要跳过的文件")
    
    # 读取CSV文件，并过滤掉需要跳过的图像
    images_to_copy = []
    skipped_from_csv = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'image_name' in row and 'full_path' in row:
                image_name = row['image_name']
                
                # 检查是否需要跳过该图像
                if image_name in skip_files:
                    skipped_from_csv += 1
                    continue
                
                images_to_copy.append({
                    'image_name': image_name,
                    'target_path': row['full_path']
                })
    
    if not images_to_copy:
        print("CSV文件中没有找到有效的图像数据")
        return
    
    print(f"从CSV文件中读取了 {len(images_to_copy)} 个图像记录")
    if skipped_from_csv > 0:
        print(f"从CSV中跳过了 {skipped_from_csv} 个图像（在跳过列表中）")
    
    # 构建源数据集中所有图像的映射
    source_image_map = {}
    print("正在扫描源数据集目录...")
    
    if recursive_search:
        # 递归搜索所有子目录
        for root, dirs, files in os.walk(source_data_dir):
            for file in files:
                # 跳过在跳过列表中的文件
                if file in skip_files:
                    continue
                    
                if file not in source_image_map:  # 只记录第一次出现的文件
                    source_image_map[file] = os.path.join(root, file)
    else:
        # 只搜索顶级目录
        for item in os.listdir(source_data_dir):
            item_path = os.path.join(source_data_dir, item)
            if os.path.isfile(item_path):
                # 跳过在跳过列表中的文件
                if item in skip_files:
                    continue
                    
                if item not in source_image_map:
                    source_image_map[item] = item_path
    
    print(f"在源数据集中找到 {len(source_image_map)} 个图像文件（已过滤跳过文件）")
    
    # 复制图像文件
    copied_count = 0
    missing_count = 0
    
    for image_info in images_to_copy:
        image_name = image_info['image_name']
        target_path = image_info['target_path']
        
        # 检查源图像是否存在
        if image_name in source_image_map:
            source_path = source_image_map[image_name]
            
            # 确保目标目录存在
            target_dir = os.path.dirname(target_path)
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                # 复制文件
                shutil.copy2(source_path, target_path)
                copied_count += 1
                print(f"已复制: {image_name} -> {target_path}")
            except Exception as e:
                print(f"复制文件 {image_name} 时出错: {e}")
        else:
            print(f"未找到源图像: {image_name}")
            missing_count += 1
    
    print(f"\n复制完成!")
    print(f"成功复制: {copied_count} 个文件")
    print(f"未找到: {missing_count} 个文件")
    print(f"跳过的文件: {len(skip_files)} 个（从跳过列表）")

def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='根据YAML配置从源数据集复制图像到指定位置')
    
    # 添加配置文件参数
    parser.add_argument('--config', type=str,default='',
                       help='YAML配置文件路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        print("请创建配置文件，格式如下:")
        print("""
csv_file: "/path/to/your/image_list.csv"
source_dir: "/path/to/source/dataset"
skip_file: "/path/to/skip/files.txt"  # 可选
recursive_search: true  # 可选，默认为true
""")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 调用函数
    copy_images_from_csv(config)

if __name__ == "__main__":
    main()