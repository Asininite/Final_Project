import sys
from pathlib import Path

# ensure project root is on sys.path so 'models' can be imported when run from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models import SmallCNN


def export_torchscript(out_path: str = 'artifacts/detector.pt'):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model = SmallCNN()
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    print(f'Wrote TorchScript artifact to {out_path}')


if __name__ == '__main__':
    export_torchscript()
