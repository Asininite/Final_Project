import sys
import json
import argparse
from pathlib import Path

# ensure project root is on sys.path so project modules can be imported when run from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from backbones import get_model_and_transforms, IMAGENET_MEAN, IMAGENET_STD


def _infer_input_size(backbone: str) -> int:
    b = (backbone or 'smallcnn').lower()
    if b == 'smallcnn':
        return 64
    if b in ('xception', 'xceptionnet'):
        return 299
    # default for imagenet cnn
    return 224


def export_torchscript(out_path: str, backbone: str, num_classes: int, ckpt_path: str | None, pretrained: bool, freeze_backbone: bool):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build model
    model, _, _ = get_model_and_transforms(backbone, num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)

    # Load checkpoint if provided
    if ckpt_path:
        ckpt_p = Path(ckpt_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        state = torch.load(ckpt_p, map_location='cpu')
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)

    model.eval()

    size = _infer_input_size(backbone)
    example = torch.randn(1, 3, size, size)
    traced = torch.jit.trace(model, example)
    traced.save(str(out))

    # Write metadata for server transforms
    meta = {
        'backbone': backbone,
        'input_size': size,
        'normalize': {
            'mean': IMAGENET_MEAN if backbone.lower() != 'smallcnn' else None,
            'std': IMAGENET_STD if backbone.lower() != 'smallcnn' else None,
        }
    }
    meta_path = out.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print(f'Wrote TorchScript artifact to {out}')
    print(f'Wrote metadata to {meta_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=str(PROJECT_ROOT / 'artifacts' / 'detector.pt'))
    parser.add_argument('--backbone', type=str, default='smallcnn')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default=str(PROJECT_ROOT / 'artifacts' / 'best.pt'))
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    args = parser.parse_args()

    export_torchscript(
        out_path=args.out,
        backbone=args.backbone,
        num_classes=args.num_classes,
        ckpt_path=args.ckpt,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
