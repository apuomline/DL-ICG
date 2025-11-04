"""
在effnet_train.py基础上使用了arceface head - EfficientNet版本
修改了学习率调度器，与代码1和代码2保持一致
支持YAML配置文件调整模型规格和训练超参数
"""

import os
import argparse
import math
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import timm
import datetime
import types
from typing import List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ============================ ArcFace Head 实现 ============================

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

    def forward(self, features: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
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

# ============================ 学习率调度器 (与代码1和代码2一致) ============================

def lr_warmup(lr_list, lr_init, warmup_end_epoch=5):
    """学习率warmup"""
    lr_list[:warmup_end_epoch] = list(np.linspace(0, lr_init, warmup_end_epoch))
    return lr_list

def lr_scheduler(lr_init, num_epochs, warmup_end_epoch=5, mode='cosine'):
    """
    学习率调度器 - 与代码1和代码2保持一致
    :param lr_init：初始学习率
    :param num_epochs: 总epoch数
    :param warmup_end_epoch: warmup的epoch数
    :param mode: {cosine}
                  cosine: lr_t = 0.5 * lr_0 * (1 + cos(t * pi / T)) 在第t个epoch
    """
    lr_list = [lr_init] * num_epochs

    print(f'*** 学习率warmup {warmup_end_epoch}个epoch')
    lr_list = lr_warmup(lr_list, lr_init, warmup_end_epoch)

    print(f'*** 学习率使用 {mode} 模式衰减')
    if mode == 'cosine':
        for t in range(warmup_end_epoch, num_epochs):
            lr_list[t] = 0.5 * lr_init * (1 + math.cos((t - warmup_end_epoch + 1) * math.pi / (num_epochs - warmup_end_epoch)))
    else:
        raise AssertionError(f'{mode} 模式未实现')
    return lr_list

def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    """调整学习率和beta参数 - 与代码1保持一致"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # 只改变beta1

# ============================ 配置管理 ============================

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将配置转换为命名空间对象
    args = types.SimpleNamespace()
    
    # 基础配置
    base_config = config.get('base', {})
    args.seed = base_config.get('seed', 42)
    args.num_classes = base_config.get('num_classes', 5000)
    args.image_size = base_config.get('image_size', 384)
    args.batch_size = base_config.get('batch_size', 64)
    args.max_device_batch_size = base_config.get('max_device_batch_size', 16)
    args.num_workers = base_config.get('num_workers', 8)
    
    # 模型配置
    model_config = config.get('model', {})
    args.model_name = model_config.get('name', 'efficientnet_b5')
    args.pretrained_path = model_config.get('pretrained_path', None)
    
    # ArcFace配置
    arcface_config = config.get('arcface', {})
    args.arcface_s = arcface_config.get('s', 30.0)
    args.arcface_m = arcface_config.get('m', 0.5)
    args.number_sub_center = arcface_config.get('number_sub_center', 1)
    
    # 优化器配置
    optimizer_config = config.get('optimizer', {})
    args.base_learning_rate = optimizer_config.get('base_learning_rate', 0.001)
    args.weight_decay = optimizer_config.get('weight_decay', 1e-8)
    args.total_epoch = optimizer_config.get('total_epoch', 200)
    args.warmup_epoch = optimizer_config.get('warmup_epoch', 5)
    
    # 数据增强配置
    augmentation_config = config.get('augmentation', {})
    args.use_randaug = augmentation_config.get('use_randaug', False)
    args.randaug_n = augmentation_config.get('randaug_n', 2)
    args.randaug_m = augmentation_config.get('randaug_m', 9)
    args.random_erasing_p = augmentation_config.get('random_erasing_p', 0.25)
    args.mixup_alpha = augmentation_config.get('mixup_alpha', 0.2)
    args.cutmix_alpha = augmentation_config.get('cutmix_alpha', 1.0)
    args.use_flip_tta = augmentation_config.get('use_flip_tta', True)
    
    # 归一化配置
    normalize_config = config.get('normalize', {})
    args.normalize_mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
    args.normalize_std = normalize_config.get('std', [0.229, 0.224, 0.225])
    
    # 路径配置
    path_config = config.get('paths', {})
    args.output_dir = path_config.get('output_dir', '/root/autodl-tmp/AIC/EfficientNet/outputs')
    args.exp_name = path_config.get('exp_name', None)
    args.resume = path_config.get('resume', None)
    args.train_dir = path_config.get('train_dir', '/root/autodl-tmp/AIC/dataset/webinat5000/train')
    args.val_dir = path_config.get('val_dir', '/root/autodl-tmp/AIC/dataset/webinat5000/val')
    
    return args, config

def save_config(config, config_path):
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

# ============================ 工具函数 ============================

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_transforms(image_size: int, mean, std, use_randaug: bool, randaug_n: int, randaug_m: int, random_erasing_p: float):
    """构建数据增强流程"""
    train_tf_list = [
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    if use_randaug:
        train_tf_list.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
    
    train_tf_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=random_erasing_p, inplace=False),
    ]
    train_tf = transforms.Compose(train_tf_list)

    eval_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf

def build_datasets(args):
    """构建数据集"""
    train_tf, eval_tf = build_transforms(
        args.image_size,
        args.normalize_mean,
        args.normalize_std,
        args.use_randaug,
        args.randaug_n,
        args.randaug_m,
        args.random_erasing_p,
    )

    print(f"Using datasets:")
    print(f"  Train directory: {args.train_dir}")
    print(f"  Val directory: {args.val_dir}")

    # 放宽文件校验
    from torchvision.datasets.folder import has_file_allowed_extension
    from PIL import Image
    ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp',
                    '.gif', '.jfif')

    def is_valid_file(path: str) -> bool:
        p = path.strip()
        if has_file_allowed_extension(p, ALLOWED_EXTS):
            return True
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    train_ds = torchvision.datasets.ImageFolder(args.train_dir, transform=train_tf, is_valid_file=is_valid_file)
    val_ds = torchvision.datasets.ImageFolder(args.val_dir, transform=eval_tf, is_valid_file=is_valid_file)

    return train_ds, val_ds

def create_efficientnet_arcface_model(model_name, num_classes, arcface_s=30.0, arcface_m=0.5, 
                                     number_sub_center=1, pretrained_path=None):
    """创建EfficientNet + ArcFace模型"""
    print(f"Creating {model_name} model with {num_classes} classes and ArcFace head")
    
    # 创建EfficientNet骨干网络
    if pretrained_path is None:
        print("Using timm's ImageNet pretrained weights")
        backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)  # 不要分类头
    else:
        print(f"Creating model without pretrained weights, will load from: {pretrained_path}")
        backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        print(f"Loading pretrained weights: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                elif k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                elif k.startswith('head.'):
                    continue  # 跳过分类头权重
                else:
                    new_state_dict[k] = v
            
            missing_keys, unexpected_keys = backbone.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                
            print(f"✅ Successfully loaded pretrained weights")
        except Exception as e:
            print(f"❌ Warning: Failed to load pretrained weights {pretrained_path}")
            print(f"Error: {str(e)}")
            print("Using randomly initialized weights")
    
    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
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
    return model

def evaluate(model, loader, device, special_label_indices=None, use_flip_tta: bool = True):
    """评估模型"""
    model.eval()
    correct = 0.0
    total = 0
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    use_special = bool(special_label_indices)
    special_tensor = None
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            if use_flip_tta:
                # 测试时增强
                logits1 = model(imgs)
                logits2 = model(torch.flip(imgs, dims=[3]))
                logits = (logits1 + logits2) / 2.0
                loss = criterion(logits, labels)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            pred = torch.argmax(logits, dim=1)
            equals = (pred == labels).float()
            
            if use_special:
                if special_tensor is None:
                    special_tensor = torch.tensor(sorted(list(special_label_indices)), device=labels.device)
                mask_special = torch.isin(labels, special_tensor)
                equals = equals.masked_fill(mask_special, 0.0)
                correct_batch = equals.sum().item() + 0.5 * mask_special.sum().item()
            else:
                correct_batch = equals.sum().item()

            correct += correct_batch
            total += labels.numel()
            losses.append(loss.item())
            
    acc = correct / max(total, 1)
    return acc, float(np.mean(losses) if losses else 0.0)

def _rand_bbox(size, lam):
    """CutMix的随机边界框生成"""
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def create_experiment_dir(base_output_dir, exp_name=None):
    """创建实验目录"""
    if exp_name:
        exp_dir = os.path.join(base_output_dir, exp_name)
    else:
        existing_dirs = glob.glob(os.path.join(base_output_dir, 'exp_*'))
        existing_nums = []
        for d in existing_dirs:
            if os.path.isdir(d):
                dir_name = os.path.basename(d)
                try:
                    num = int(dir_name.split('_')[1])
                    existing_nums.append(num)
                except (IndexError, ValueError):
                    continue
        
        next_num = max(existing_nums) + 1 if existing_nums else 1
        exp_dir = os.path.join(base_output_dir, f'exp_{next_num:03d}')
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f'Experiment directory: {os.path.abspath(exp_dir)}')
    return exp_dir

def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """加载检查点"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded")
    
    # 加载学习率调度器状态
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("Learning rate scheduler state loaded")
    
    # 获取起始epoch和最佳准确率
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
    
    return start_epoch, best_acc

def write_epoch_result(log_file, epoch, train_loss, val_loss, val_acc, learning_rate, is_best=False):
    """将epoch结果写入日志文件"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_marker = " [BEST]" if is_best else ""
    
    log_entry = (
        f"[{timestamp}] Epoch: {epoch:03d}, "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {val_loss:.6f}, "
        f"Val Acc: {val_acc:.4f}, "
        f"LR: {learning_rate:.8f}"
        f"{best_marker}\n"
    )
    
    log_file.write(log_entry)
    log_file.flush()

# ============================ 训练函数 ============================

def train_model(args, config):
    """训练模型"""
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建实验目录
    exp_dir = create_experiment_dir(args.output_dir, args.exp_name)
    print(f'Experiment directory: {os.path.abspath(exp_dir)}')
    
    # 保存参数和配置
    args_path = os.path.join(exp_dir, 'args.txt')
    with open(args_path, 'w', encoding='utf-8') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    # 保存YAML配置
    config_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_path)
    
    # 创建训练日志文件
    log_file_path = os.path.join(exp_dir, 'training_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # 写入日志文件头部信息
    log_file.write("Training Log - EfficientNet + ArcFace\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Model: {args.model_name}\n")
    log_file.write(f"Number of classes: {args.num_classes}\n")
    log_file.write(f"Image size: {args.image_size}\n")
    log_file.write(f"ArcFace s: {args.arcface_s}, m: {args.arcface_m}\n")
    log_file.write(f"Batch size: {args.batch_size}\n")
    log_file.write(f"Total epochs: {args.total_epoch}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write("Epoch Results:\n")
    log_file.write("-" * 80 + "\n")
    log_file.flush()
    
    # 数据集
    print(f'Building datasets...')
    train_ds, val_ds = build_datasets(args)
    print(f'Train dataset size: {len(train_ds)}')
    print(f'Val dataset size: {len(val_ds)}')
    print(f'Number of classes: {len(train_ds.classes)}')
    
    load_bs = min(args.max_device_batch_size, args.batch_size)
    assert args.batch_size % load_bs == 0
    steps_per_update = args.batch_size // load_bs
    print(f'Batch size: {args.batch_size}, Load batch size: {load_bs}, Steps per update: {steps_per_update}')
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=load_bs, shuffle=True, 
        num_workers=args.num_workers, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=load_bs, shuffle=False, 
        num_workers=args.num_workers, pin_memory=False
    )
    print(f'Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}')

    # 创建EfficientNet + ArcFace模型
    model = create_efficientnet_arcface_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        number_sub_center=args.number_sub_center,
        pretrained_path=args.pretrained_path
    ).to(device)
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
    else:
        print(f'Using single device: {device}')

    # ============================ 学习率调度设置 (与代码1和代码2一致) ============================
    
    # 计算初始学习率
    use_pretrained = args.pretrained_path is not None
    base_lr = args.base_learning_rate * 2 if not use_pretrained else args.base_learning_rate
    initial_lr = base_lr * args.batch_size / 256
    
    # 创建学习率计划和beta1计划
    alpha_plan = lr_scheduler(initial_lr, args.total_epoch, warmup_end_epoch=args.warmup_epoch, mode='cosine')
    
    # Beta1参数调整 (与代码1一致)
    mom1 = 0.9
    mom2 = 0.1
    epoch_decay_start = 40  # 与代码1保持一致
    
    beta1_plan = [mom1] * args.total_epoch
    for i in range(epoch_decay_start, args.total_epoch):
        beta1_plan[i] = mom2
    
    print(f"Learning rate schedule: warmup={args.warmup_epoch} epochs, cosine decay")
    print(f"Beta1 schedule: {mom1} for first {epoch_decay_start} epochs, then {mom2}")

    # 优化器 - 使用与代码1相同的参数
    optim = torch.optim.Adam(
        model.parameters(), 
        lr=initial_lr,  # 学习率会在每个epoch调整
        betas=(mom1, 0.999),  # 初始beta值，会在训练中调整
        weight_decay=args.weight_decay
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # TensorBoard日志
    dataset_name = os.path.basename(os.path.abspath(args.train_dir)) or 'dataset'
    log_dir = os.path.join(exp_dir, 'logs', dataset_name)
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard logs: {os.path.abspath(log_dir)}')

    # 初始化训练变量
    start_epoch = 0
    best_acc = 0.0
    step_count = 0
    mixup_alpha = args.mixup_alpha
    cutmix_alpha = args.cutmix_alpha
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0
    
    # 恢复训练
    if args.resume:
        try:
            start_epoch, best_acc = load_checkpoint(
                args.resume, model, optim, None  # 不使用LambdaLR调度器
            )
            print(f"Resumed training from epoch {start_epoch}")
            log_file.write(f"[INFO] Resumed from checkpoint: {args.resume}\n")
            log_file.write(f"[INFO] Starting from epoch {start_epoch}, best accuracy so far: {best_acc:.4f}\n")
            log_file.flush()
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0
            best_acc = 0.0
            log_file.write(f"[WARNING] Failed to load checkpoint: {e}\n")
            log_file.write("[INFO] Starting training from scratch\n")
            log_file.flush()

    # 训练循环
    optim.zero_grad()
    
    for e in range(start_epoch, args.total_epoch):
        # 调整学习率和beta参数 (与代码1一致)
        adjust_learning_rate(optim, e, alpha_plan, beta1_plan)
        current_lr = optim.param_groups[0]['lr']
        current_beta1 = optim.param_groups[0]['betas'][0]
        
        model.train()
        losses = []
        with tqdm(total=len(train_loader), 
                  desc=f'Epoch {e+1}/{args.total_epoch} (LR: {current_lr:.2e}, β1: {current_beta1:.3f})', 
                  mininterval=0.3) as pbar:
            for imgs, labels in train_loader:
                step_count += 1
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # Mixup / CutMix
                if use_mixup or use_cutmix:
                    lam = 1.0
                    rand = torch.rand(1).item()
                    if use_cutmix and (not use_mixup or rand < 0.5):
                        # CutMix
                        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                        batch_size = imgs.size(0)
                        index = torch.randperm(batch_size, device=device)
                        bbx1, bby1, bbx2, bby2 = _rand_bbox(imgs.size(), lam)
                        imgs = imgs.clone()
                        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[index, :, bby1:bby2, bbx1:bbx2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
                        # 使用ArcFace前向传播
                        logits = model(imgs, labels)
                        loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[index])
                    else:
                        # Mixup
                        lam = np.random.beta(mixup_alpha, mixup_alpha)
                        batch_size = imgs.size(0)
                        index = torch.randperm(batch_size, device=device)
                        mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
                        # 使用ArcFace前向传播
                        logits = model(mixed_x, labels)
                        loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[index])
                else:
                    # 标准训练，使用ArcFace前向传播
                    logits = model(imgs, labels)
                    loss = criterion(logits, labels)
                    
                loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
                pbar.set_postfix(**{'Loss': np.mean(losses)})
                pbar.update(1)

        train_loss = float(np.mean(losses) if losses else 0.0)
        
        # 验证阶段
        special_classes = {3297, 3942}  # 特殊类别处理
        val_acc, val_loss = evaluate(
            model, val_loader, device,
            special_label_indices=special_classes,
            use_flip_tta=args.use_flip_tta
        )
        
        writer.add_scalar('train_loss', train_loss, global_step=e)
        writer.add_scalar('val_loss', val_loss, global_step=e)
        writer.add_scalar('val_acc', val_acc, global_step=e)
        writer.add_scalar('learning_rate', current_lr, global_step=e)
        writer.add_scalar('beta1', current_beta1, global_step=e)

        # 检查是否为最佳模型
        is_best = val_acc > best_acc
        
        # 将epoch结果写入日志文件
        write_epoch_result(log_file, e+1, train_loss, val_loss, val_acc, current_lr, is_best)

        # 保存检查点
        checkpoint = {
            'epoch': e,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optim.state_dict(),
            'best_acc': best_acc,
            'args': vars(args)
        }
        checkpoint_path = os.path.join(exp_dir, 'checkpoint.pth')
        save_checkpoint(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_acc = val_acc
            best_state = model.state_dict()
            best_model_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save(best_state, best_model_path)
            print(f'[BEST] val_acc={best_acc:.4f}, saved to {best_model_path}')
            log_file.write(f"[BEST] Updated best model with accuracy: {best_acc:.4f}\n")
            log_file.flush()

    writer.close()
    
    # 保存最终权重
    final_state_path = os.path.join(exp_dir, f'{args.model_name}_final.pth')
    torch.save(model.state_dict(), final_state_path)
    
    # 写入训练结束信息
    log_file.write("-" * 80 + "\n")
    log_file.write(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Final best accuracy: {best_acc:.4f}\n")
    log_file.close()
    
    # 保存训练摘要
    summary = {
        'best_val_acc': best_acc,
        'final_epoch': args.total_epoch,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'arcface_s': args.arcface_s,
        'arcface_m': args.arcface_m,
        'learning_rate': base_lr,
        'weight_decay': args.weight_decay,
        'model_name': args.model_name,
        'experiment_dir': exp_dir,
    }
    
    summary_path = os.path.join(exp_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Training Summary - EfficientNet + ArcFace\n')
        f.write('=' * 50 + '\n')
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')
    
    print(f'[DONE] Final model saved to: {final_state_path}')
    print(f'[DONE] Best model saved to: {best_model_path}')
    print(f'[DONE] Training summary saved to: {summary_path}')
    print(f'[DONE] Training log saved to: {log_file_path}')
    
    return best_acc, exp_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNet + ArcFace for fine-grained classification')
    parser.add_argument('--config', type=str, default='./configs/final_test_config_web400.yaml', help='Path to the configuration YAML file')
    
    args_cmd = parser.parse_args()
    
    # 加载配置文件
    args, config = load_config(args_cmd.config)
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 验证数据集目录
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory does not exist: {args.train_dir}")
        exit(1)
    
    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory does not exist: {args.val_dir}")
        exit(1)
    
    # 创建基础输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Base output directory: {os.path.abspath(args.output_dir)}')
    
    # 训练模型
    print(f"Starting training with EfficientNet + ArcFace")
    print(f"ArcFace parameters: s={args.arcface_s}, m={args.arcface_m}")
    print(f"Learning rate schedule: warmup={args.warmup_epoch} epochs + cosine decay")
    print(f"Weight decay: {args.weight_decay}")
    
    if args.pretrained_path:
        print(f"Using custom pretrained weights from: {args.pretrained_path}")
    else:
        print("Using timm's ImageNet pretrained weights")
    
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
    
    best_acc, exp_dir = train_model(args, config)
    print(f"Training completed with best accuracy: {best_acc:.4f}")
    print(f'[DONE] All outputs saved to: {os.path.abspath(exp_dir)}')