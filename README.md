Adversarial Deepfake Detector (prototype)

This repository is a minimal PyTorch prototype that implements a detector for adversarially attacked deepfakes.

Contents
- `train.py` — training and evaluation entrypoint
- `datasets.py` — synthetic dataset that generates clean and attacked "deepfake" images for quick testing
- `models.py` — small CNN detector
- `attacks.py` — FGSM and PGD attack implementations
- `losses.py` — AFSL loss (classification + consistency + feature similarity)
- `utils.py` — helper functions
- `requirements.txt` — Python dependencies

Quickstart (Windows PowerShell)

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Run a short smoke training (1 epoch):

```powershell
python train.py --epochs 1 --batch-size 32 --smoke
```

AFSL training (feature similarity + adversarial consistency)

The repo supports an AFSL-style objective that combines standard classification with adversarial prediction consistency and feature similarity between clean/attacked pairs. Enable via flags:

```powershell
# Enable AFSL with FGSM pairs and cosine similarity
python train.py --epochs 1 --batch-size 32 --smoke --use-afsl --afsl-attack fgsm --lambda-adv 1.0 --lambda-sim 1.0 --sim-metric cosine

# Or with PGD pairs
python train.py --epochs 1 --batch-size 32 --smoke --use-afsl --afsl-attack pgd --pgd-steps 5 --pgd-alpha 0.01
```

You'll see AFSL component summaries per epoch like:

```
AFSL(l_cls=..., l_adv=..., l_sim=...)
```

Small-scale FF++ demo (frames/crops)

1) Preprocess a tiny set of videos into face crops and a catalog CSV:

```powershell
python tools/face_preprocess.py --video-dir path\to\small_video_set --out-dir data\ffpp\aligned --fps 1
```

This writes `data/ffpp/aligned/catalog.csv` with columns `path,label,video`.

2) Train on that catalog with a pretrained backbone and evaluate adversarial robustness:

```powershell
python train.py --dataset-csv data\ffpp\aligned\catalog.csv --backbone resnet50 --pretrained --epochs 1 --batch-size 16 --adv-eval fgsm --smoke
```

Useful flags:
- `--backbone {smallcnn,resnet50,mobilenet_v3_small,mobilenet_v3_large}` (add `--pretrained` for ImageNet weights)
- `--backbone xception` via timm (ImageNet-pretrained), uses 299×299 transforms
- `--adv-train {none,fgsm,pgd}` with `--eps`, `--pgd-steps`, `--pgd-alpha`
- AFSL: `--use-afsl`, `--afsl-attack {fgsm,pgd}`, `--lambda-adv`, `--lambda-sim`, `--sim-metric {cosine,mse}`
- `--save-dir artifacts` and `--resume artifacts\\last.pt`

Notes
- This prototype ships with a synthetic dataset for smoke tests. For real data, use the FF++ loader (`datasets/ffpp_dataset.py`) via `--dataset-csv` as shown above.
- Attacks (`attacks.py`) include FGSM, PGD, and C&W (slower) for evaluation/training.
- Checkpoints are saved to `artifacts/last.pt` and `artifacts/best.pt`.
 - AFSL features are captured using a forward hook on the final classifier layer for torchvision backbones; no code changes needed.
	- Xception backbone requires `timm`; it's included in `requirements.txt`.