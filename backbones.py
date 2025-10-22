import torch.nn as nn
from torchvision import models, transforms
try:
    import timm
except Exception:
    timm = None
from utils import attach_penultimate_feature_hook


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _imagenet_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _smallcnn_transforms(train: bool, size: int = 64):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])


def resnet50_backbone(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    # Using deprecated pretrained flag for broad compatibility
    model = models.resnet50(pretrained=pretrained)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _xception_transforms(train: bool):
    # Xception is typically trained at 299x299 with ImageNet normalization
    size = 299
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def xception_backbone(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
    if timm is None:
        raise ImportError("timm is required for xception backbone. Please install timm.")
    # Create model with desired num_classes
    model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze typical classifier params if named accordingly
        for name, p in model.named_parameters():
            if any(k in name for k in ('classifier', 'fc', 'head')):
                p.requires_grad = True
    return model


def mobilenet_v3_backbone(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False, variant: str = 'large') -> nn.Module:
    if variant == 'large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
    else:
        model = models.mobilenet_v3_small(pretrained=pretrained)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # Replace classifier head
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        # Fallback
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    return model


def get_model_and_transforms(name: str, num_classes: int = 2, pretrained: bool = False, freeze_backbone: bool = False):
    name = (name or 'smallcnn').lower()
    if name == 'resnet50':
        model = resnet50_backbone(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
        train_tf, val_tf = _imagenet_transforms(train=True), _imagenet_transforms(train=False)
    elif name in ('mobilenet_v3_large', 'mobilenetv3_large', 'mobilenetv3'):
        model = mobilenet_v3_backbone(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone, variant='large')
        train_tf, val_tf = _imagenet_transforms(train=True), _imagenet_transforms(train=False)
    elif name in ('mobilenet_v3_small', 'mobilenetv3_small'):
        model = mobilenet_v3_backbone(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone, variant='small')
        train_tf, val_tf = _imagenet_transforms(train=True), _imagenet_transforms(train=False)
    elif name == 'smallcnn':
        from models import SmallCNN
        model = SmallCNN(num_classes=num_classes)
        train_tf, val_tf = _smallcnn_transforms(train=True, size=64), _smallcnn_transforms(train=False, size=64)
    elif name in ('xception', 'xceptionnet'):
        model = xception_backbone(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
        train_tf, val_tf = _xception_transforms(train=True), _xception_transforms(train=False)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    # Attach a feature hook for AFSL to capture penultimate features
    try:
        attach_penultimate_feature_hook(model)
    except Exception:
        pass
    return model, train_tf, val_tf
