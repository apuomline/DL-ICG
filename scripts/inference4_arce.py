"""
tta+类别平衡后处理 - 支持400类别和5000类别 - 修复版本（支持ArcFace模型）
简化版本：移除类别平衡代码，使用YAML配置
"""

import os
import yaml
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import timm
import glob
from tqdm import tqdm
import json
import csv
from collections import defaultdict
import math
import torch.nn as nn
import torch.nn.functional as F

# ============================ ArcFace Head 实现（与训练代码一致） ============================

class NormLinear(nn.Linear):
    """归一化线性层"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 feature_norm: bool = True,
                 weight_norm: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)

class SubCenterNormLinear(nn.Linear):
    """子中心归一化线性层"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 k=3,
                 feature_norm: bool = True,
                 weight_norm: bool = True):
        super().__init__(in_features, out_features * k, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm
        self.out_features = out_features
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        cosine_all = F.linear(input, weight, self.bias)
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

class ArcFaceHead(nn.Module):
    """ArcFace分类头"""
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 s: float = 30.0,
                 m: float = 0.50,
                 number_sub_center: int = 1,
                 easy_margin: bool = False,
                 ls_eps: float = 0.0,
                 bias: bool = False):
        super(ArcFaceHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps

        if self.num_classes <= 0:
            raise ValueError(f'num_classes={num_classes} must be a positive integer')

        self.easy_margin = easy_margin
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        assert number_sub_center >= 1
        if number_sub_center == 1:
            self.norm_linear = NormLinear(in_channels, num_classes, bias=bias)
        else:
            self.norm_linear = SubCenterNormLinear(
                in_channels, num_classes, bias=bias, k=number_sub_center)

    def forward(self, features: torch.Tensor, target = None) -> torch.Tensor:
        """前向传播"""
        # 确保在FP32精度下计算
        features = features.float()
        
        # cos=(a*b)/(||a||*||b||)
        cosine = self.norm_linear(features)

        if target is None:
            # 测试阶段，直接返回cosine乘scale
            return self.s * cosine

        # 训练阶段，应用ArcFace边际
        phi = torch.cos(torch.acos(cosine) + self.m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 创建one-hot编码
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.num_classes

        # 组合输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

# ============================ 模型创建函数（与训练代码一致） ============================

def create_efficientnet_arcface_model(model_name, num_classes, checkpoint_path, arcface_s=30.0, arcface_m=0.5, number_sub_center=1):
    """创建EfficientNet + ArcFace模型（与训练代码一致）"""
    print(f"Creating {model_name} model with {num_classes} classes and ArcFace head")
    
    # 创建EfficientNet骨干网络
    print(f"Creating model without pretrained weights, will load from: {checkpoint_path}")
    backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 384, 384)  # 使用固定的image_size
        features = backbone(dummy_input)
        feature_dim = features.shape[1]
        print(f"Feature dimension: {feature_dim}")
    
    # 创建ArcFace分类头
    arcface_head = ArcFaceHead(
        num_classes=num_classes,
        in_channels=feature_dim,
        s=arcface_s,
        m=arcface_m,
        number_sub_center=number_sub_center,
        easy_margin=False,
        ls_eps=0.0,
        bias=False
    )
    
    # 组合成完整模型
    class EfficientNetArcFaceModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
            
        def forward(self, x, target=None):
            features = self.backbone(x)
            return self.head(features, target)
    
    model = EfficientNetArcFaceModel(backbone, arcface_head)
    
    # 加载训练好的权重
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的state_dict格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理权重键名
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除可能存在的module.前缀
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
    
    if missing_keys:
        print(f"⚠️ 缺失的键: {len(missing_keys)} 个")
        for i, key in enumerate(missing_keys[:10]):
            print(f"  {i+1}. {key}")
        if len(missing_keys) > 10:
            print(f"  ... 还有 {len(missing_keys) - 10} 个缺失的键")
    
    if unexpected_keys:
        print(f"⚠️ 意外的键: {len(unexpected_keys)} 个")
        for i, key in enumerate(unexpected_keys[:10]):
            print(f"  {i+1}. {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... 还有 {len(unexpected_keys) - 10} 个意外的键")
    
    print(f"✅ 成功加载模型权重")
    return model

# ============================ 其他函数保持不变 ============================

def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_test_transform(image_size: int, mean, std):
    """构建测试时的数据预处理流程"""
    test_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return test_tf

def load_image_paths(test_dir):
    """加载测试目录下的所有图像路径"""
    # 支持的图像格式
    ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp',
                    '.gif', '.jfif')
    
    image_paths = []
    for ext in ALLOWED_EXTS:
        pattern = os.path.join(test_dir, f'*{ext}')
        image_paths.extend(glob.glob(pattern))
        pattern = os.path.join(test_dir, f'*{ext.upper()}')
        image_paths.extend(glob.glob(pattern))
    
    # 去重并排序
    image_paths = sorted(list(set(image_paths)))
    print(f"找到 {len(image_paths)} 张测试图像")
    return image_paths

def tta_forward(model, inputs):
    """TTA前向传播"""
    # 原始预测
    ori_out = model(inputs)
    # 水平翻转预测
    flip_out = model(inputs.flip(3))  # 在维度3（宽度）上翻转
    # 取平均
    out = (ori_out + flip_out) / 2
    return out

def predict_single_image(model, image_path, transform, device, use_tta=True):
    """对单张图像进行预测"""
    try:
        # 加载图像
        image = Image.open(image_path)
        
        # 应用预处理
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            if use_tta:
                # 使用TTA（水平翻转增强）
                logits = tta_forward(model, input_tensor)
            else:
                logits = model(input_tensor)  # 注意：这里不需要target参数
            
            # 计算概率和预测类别
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            full_probabilities = probabilities.cpu().numpy()[0]  # 获取完整概率向量
            
        return predicted_class, confidence, full_probabilities, True
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return -1, 0.0, None, False

def predict_batch_images(model, image_paths, transform, device, batch_size=32, use_tta=True):
    """批量预测图像"""
    results = []
    all_probabilities = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="推理进度"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []
        
        # 加载和预处理批次图像
        for j, img_path in enumerate(batch_paths):
            try:
                image = Image.open(img_path)
                input_tensor = transform(image)
                batch_images.append(input_tensor)
                valid_indices.append(j)
            except Exception as e:
                print(f"加载图像 {img_path} 失败: {e}")
                continue
        
        if not batch_images:
            continue
            
        # 堆叠张量
        batch_tensor = torch.stack(batch_images).to(device)
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            if use_tta:
                # 使用TTA（水平翻转增强）
                logits = tta_forward(model, batch_tensor)
            else:
                logits = model(batch_tensor)  # 注意：这里不需要target参数
            
            # 计算概率和预测类别
            probabilities = torch.softmax(logits, dim=1)
            confidences, predicted_classes = torch.max(probabilities, 1)
            
            # 收集结果
            batch_probs = probabilities.cpu().numpy()
            for idx, (img_idx, pred_class, conf) in enumerate(zip(valid_indices, predicted_classes, confidences)):
                original_idx = i + img_idx
                results.append({
                    'image_path': batch_paths[img_idx],
                    'predicted_class': pred_class.item(),
                    'confidence': conf.item(),
                    'status': 'success'
                })
                all_probabilities.append(batch_probs[idx])
    
    # 处理失败的情况
    success_paths = {r['image_path'] for r in results}
    for img_path in image_paths:
        if img_path not in success_paths:
            results.append({
                'image_path': img_path,
                'predicted_class': -1,
                'confidence': 0.0,
                'status': 'failed'
            })
            # 对于失败的图像，添加均匀分布的概率
            if all_probabilities:  # 确保至少有一个成功的概率向量
                all_probabilities.append(np.ones(len(all_probabilities[0])) / len(all_probabilities[0]))
            else:
                # 如果完全没有成功的推理，创建一个默认的概率向量
                all_probabilities.append(np.ones(5000) / 5000)  # 假设5000类
    
    return results, all_probabilities

def save_submission_csv(results, output_file='submission.csv', num_classes=400):
    """保存预测结果为CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for result in results:
            # 获取图片文件名
            image_filename = os.path.basename(result['image_path'])
            
            # 获取预测类别，如果是失败的情况默认设为0000或00000
            if result['status'] == 'success':
                predicted_class = result['predicted_class']
                # 根据类别数量决定格式
                if num_classes <= 400:
                    class_str = f"{predicted_class:04d}"
                else:
                    class_str = f"{predicted_class:05d}"
            else:
                # 预测失败时使用默认类别
                if num_classes <= 400:
                    class_str = "0000"
                else:
                    class_str = "00000"
            
            # 写入CSV文件，格式：文件名, 类别
            writer.writerow([image_filename, class_str])
    
    print(f"✅ 提交文件已生成: {output_file}")
    print(f"📊 总图像数量: {len(results)}")

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的EfficientNet+ArcFace模型进行图像分类推理（YAML配置版本）')
    parser.add_argument('--config', type=str, required=True, help='YAML配置文件路径')
    parser.add_argument('--test_dir', type=str, help='测试目录路径（覆盖配置文件中的设置）')
    parser.add_argument('--checkpoint_path', type=str, help='模型权重路径（覆盖配置文件中的设置）')
    parser.add_argument('--output_file', type=str, help='输出文件路径（覆盖配置文件中的设置）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    config = load_config(args.config)
    
    # 使用命令行参数覆盖配置文件中的设置
    if args.test_dir:
        config['test_dir'] = args.test_dir
    if args.checkpoint_path:
        config['checkpoint_path'] = args.checkpoint_path
    if args.output_file:
        config['output_file'] = args.output_file
    
    # 设置随机种子
    setup_seed(config.get('seed', 42))
    
    # 检查输入目录
    if not os.path.exists(config['test_dir']):
        print(f"错误: 测试目录不存在: {config['test_dir']}")
        return
    
    if not os.path.exists(config['checkpoint_path']):
        print(f"错误: 模型权重文件不存在: {config['checkpoint_path']}")
        return
    
    # 设置设备
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")
    
    # 构建数据预处理
    normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
    
    test_transform = build_test_transform(config['image_size'], normalize_mean, normalize_std)
    
    # 创建模型并加载权重（使用ArcFace版本）
    model = create_efficientnet_arcface_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        checkpoint_path=config['checkpoint_path'],
        arcface_s=config.get('arcface_s', 30.0),
        arcface_m=config.get('arcface_m', 0.5),
        number_sub_center=config.get('number_sub_center', 1)
    )
    model = model.to(device)
    
    # 如果使用多GPU，需要包装
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"使用DataParallel，GPU数量: {torch.cuda.device_count()}")
    
    # 加载测试图像
    image_paths = load_image_paths(config['test_dir'])
    if not image_paths:
        print("在测试目录中未找到任何图像文件")
        return
    
    # 进行推理
    use_tta = config.get('use_tta', True)
    batch_size = config.get('batch_size', 32)
    
    print(f"开始推理，使用{'TTA' if use_tta else '无TTA'}，批次大小: {batch_size}")
    
    if batch_size == 1:
        # 单张图像推理
        results = []
        all_probabilities = []
        for img_path in tqdm(image_paths, desc="推理进度"):
            pred_class, confidence, probabilities, success = predict_single_image(
                model, img_path, test_transform, device, use_tta
            )
            results.append({
                'image_path': img_path,
                'predicted_class': pred_class,
                'confidence': confidence,
                'status': 'success' if success else 'failed'
            })
            if success:
                all_probabilities.append(probabilities)
            else:
                all_probabilities.append(np.ones(config['num_classes']) / config['num_classes'])
    else:
        # 批量推理
        results, all_probabilities = predict_batch_images(
            model, image_paths, test_transform, device, batch_size, use_tta
        )
    
    # 保存推理结果
    save_submission_csv(results, config['output_file'], config['num_classes'])
    
    # 打印结果统计
    successful = [r for r in results if r['status'] == 'success']
    print(f"\n推理完成!")
    print(f"总图像: {len(results)}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(results) - len(successful)}")
    
    if successful:
        avg_conf = np.mean([r['confidence'] for r in successful])
        print(f"平均置信度: {avg_conf:.4f}")

if __name__ == '__main__':
    main()