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


class PenultimateFeatureHook:
    """Forward hook to capture the penultimate features (inputs to final Linear layer).

    Usage:
        hook = PenultimateFeatureHook(layer)
        logits = model(x)  # hook.features set to input of `layer`
        feats = hook.features
    """

    def __init__(self, layer: nn.Module):
        self.features = None

        def _hook(_module, inputs, _output):
            # inputs is a tuple; we want the tensor fed into the Linear
            if inputs and torch.is_tensor(inputs[0]):
                self.features = inputs[0].detach()
            elif inputs:
                # Some models may pass a list/tuple of tensors
                try:
                    self.features = inputs[0].detach()
                except Exception:
                    self.features = None
            else:
                self.features = None

        self.h = layer.register_forward_hook(_hook)

    def close(self):
        try:
            self.h.remove()
        except Exception:
            pass


def attach_penultimate_feature_hook(model: nn.Module) -> PenultimateFeatureHook | None:
    """Attach a forward hook to capture features before the final classifier layer.

    Supports common torchvision models (resnet*.fc, mobilenet_v3*.classifier[-1])
    and simple custom models exposing `.classifier[-1]`.
    Returns the hook object or None if a suitable layer is not found.
    """
    layer = None
    # ResNet-style
    if hasattr(model, 'fc') and isinstance(getattr(model, 'fc'), nn.Linear):
        layer = model.fc
    # MobileNet / others with classifier sequence
    elif hasattr(model, 'classifier') and isinstance(getattr(model, 'classifier'), nn.Sequential):
        seq = getattr(model, 'classifier')
        # Find last Linear in classifier
        cand = None
        for m in reversed(seq):
            if isinstance(m, nn.Linear):
                cand = m
                break
        layer = cand
    # Fallback: check for attribute commonly used in simple models
    if layer is None:
        # Scan children to find last Linear
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        layer = last_linear

    if layer is None:
        return None

    hook = PenultimateFeatureHook(layer)
    # Stash on model for convenience
    setattr(model, '_penultimate_hook', hook)
    return hook


def get_captured_features(model: nn.Module) -> torch.Tensor | None:
    """Return features captured by attach_penultimate_feature_hook for the last forward pass."""
    hook = getattr(model, '_penultimate_hook', None)
    return None if hook is None else hook.features
