import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device):
    imgs, labels = batch
    imgs = imgs.to(device)
    # Accept labels as tensor or array/scalar
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long, device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
    return imgs, labels
