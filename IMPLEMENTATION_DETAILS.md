# Implementation details: Adversarial Deepfake Detector prototype

This document describes in depth what was implemented in this prototype, step-by-step, and provides guided instructions to adapt the project to work with real deepfake datasets (FaceForensics++, Celeb-DF, etc.). It explains design decisions, pitfalls, and concrete code-level notes you can follow.

## High-level goal

Implement a machine-learning project to detect adversarially attacked deepfakes. The prototype demonstrates the pipeline: data generation, model, attacks, training & evaluation, and an example robustness evaluation using FGSM.

## Files created and purpose

- `train.py` — training and evaluation script and CLI. Implements training loop, validation, and robust evaluation using attacks.
- `datasets.py` — contains `SyntheticDeepfakeDataset`, a small dataset for quick prototyping (random images and simple synthesized "fake" samples). Also supports applying an `attack_fn` to generate adversarial versions.
- `models.py` — `SmallCNN` classifier used as detector.
- `attacks.py` — FGSM and PGD attack implementations that craft adversarial examples against the detector model.
- `utils.py` — helper utilities (parameter counting, device movement and label handling).
- `requirements.txt` — project dependencies.
- `README.md` — short quickstart and notes.
- `IMPLEMENTATION_DETAILS.md` — this file: deep explanation and guided steps to use a real dataset.

## Development steps (what I did, in order)

1. Project skeleton
   - Created the repository files listed above. Focus was on minimal, readable code that can run as a proof-of-concept.

2. Implemented the synthetic dataset (`datasets.py`)
   - The goal was to allow a smoke test without downloading large datasets.
   - `SyntheticDeepfakeDataset` returns a pair `(img, label)` with `img` a `torch.FloatTensor` in shape `(3,H,W)` and `label` an int (0 real, 1 fake).
   - Logic:
     - Base image is random noise in [0,1].
     - With 50% probability, create a synthesized fake by blending in a smooth Gaussian-like blob.
     - Optionally, if an `attack_fn` is provided and a random draw occurs, craft an adversarial image by calling `attack_fn(model, img_batch, labels)`.
   - Reasoning: keeps dataset small and deterministic-enough for debugging. Replacing this with a real dataset only requires implementing a Dataset that yields images/preprocessing identical to the current tensors.

3. Implemented the detector model (`models.py`)
   - `SmallCNN` with 3 conv layers and an MLP head. Reasoning: small models keep runtime short for prototype, while still exercising training and attack code paths.

4. Implemented attacks (`attacks.py`)
   - `fgsm_attack(model, images, labels, epsilon=0.03)` — basic one-step attack using sign of gradient.
   - `pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=10)` — projected gradient descent.
   - Important detail: attacks require gradients with respect to the inputs. So the attack call must be made while gradients are enabled; evaluation uses `torch.no_grad()` only when not crafting attacks.

5. Implemented utilities (`utils.py`)
   - `count_parameters` reports trainable parameter count.
   - `to_device` moves images and labels to the device and coerces labels to proper long tensors. This prevented a runtime issue where labels were not a tensor and attacks couldn't run.

6. Implemented training and evaluation (`train.py`)
   - `train_epoch` runs training with gradient updates.
   - `eval_epoch` supports generating adversarial examples via `attack_fn` before inference.
   - CLI: `--epochs`, `--batch-size`, and `--smoke`. `--smoke` reduces dataset sizes for quick runs.

7. Ran smoke test
   - Installed dependencies into the workspace venv.
   - Fixed a runtime bug where adversarial generation was attempted under `torch.no_grad()` (caused "does not require grad" error). Moved the attack generation out of `no_grad` and kept inference under `no_grad` afterwards.
   - Re-ran and confirmed successful training and FGSM evaluation.

## Important code-level notes and pitfalls

- Attack gradient requirements: If you compute gradients wrt input images, the tensors must require gradients and must be inside a context where autograd is enabled. In early iteration I generated attacks while inside `torch.no_grad()`, which caused a RuntimeError. Fix: call attack function outside `no_grad`, then evaluate the model under `no_grad` for speed.

- Label handling: When moving labels to device, ensure they're `torch.LongTensor`. Creating them with `torch.tensor(labels, dtype=torch.long, device=device)` fixes both list/NumPy array and tensor inputs.

- Clamping images: After adding perturbations, always clamp to valid range (here [0,1]) to avoid invalid pixel ranges.

- Determinism: This prototype uses randomness widely (synthetic data). For reproducible runs add `torch.manual_seed` and control NumPy RNG.

- Performance: This prototype is not optimized; for larger datasets use data loaders with `num_workers>0`, larger batch size, and a stronger model (ResNet/efficientnet) plus GPUs.

## How to adapt this prototype to a real deepfake dataset

Below are step-by-step guided instructions to replace the synthetic data with a real dataset and run full experiments. I'll cover:
- dataset acquisition and preprocessing
- integrating dataset into the project
- model choices and pretraining
- training recipe and adversarial training
- evaluation and metrics

### 1) Choose dataset(s)

Common public deepfake datasets:
- FaceForensics++ (FF++): contains real and manipulated videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures). Good for frame-level detectors.
- Celeb-DF: higher-quality deepfake videos with more realistic forgeries.
- DFDC (DeepFake Detection Challenge) dataset: large-scale dataset from Kaggle.

Pick one (FF++ is a pragmatic starting point).

### 2) Prepare dataset (download + filesystem layout)

Organize data on disk, e.g.:
```
data/
  ffpp/
    videos/
    frames/
    labels.csv  # optional
```

For detectors you often want per-frame images. Recommended approach:
- Extract frames from videos at a reasonable fps (1-5 fps depending on storage). Use `ffmpeg`.
- Face crop frames around detected faces so the model focuses on facial regions (use dlib or MTCNN).

Frame extraction example (ffmpeg):
```powershell
# Extract 1 fps from video.mp4 to frames dir
ffmpeg -i video.mp4 -vf fps=1 data/ffpp/frames/video_name_%05d.png
```

Face cropping pipeline (Python pseudocode):
- Use MTCNN or a face detector to find bounding boxes per frame
- Crop & optionally align faces
- Resize to desired model input resolution (e.g., 224x224)

I recommend saving processed frames to disk and building a CSV mapping filenames to label (0 real, 1 fake).

### 3) Implement a dataset class

Create `datasets/ffpp_dataset.py` (or extend `datasets.py`): implement PyTorch `Dataset` that:
- Reads image path and label from CSV or directory structure
- Loads the image (PIL), converts to RGB, normalizes and returns a `FloatTensor` in `[0,1]` (or normalized mean/std if using pretrained models)
- For training, implement augmentations (random flip, color jitter) and consider temporal sampling for video-based models

Example structure:
```python
class FFPPFramesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        # csv: filename,label
        self.entries = read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path,label = self.entries[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)
```

Transform example using torchvision:
```python
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

### 4) Model selection

Options:
- Use a pretrained backbone (ResNet50, EfficientNet) with a small classifier head for fine-tuning. This improves sample efficiency.
- For robust detectors, consider models that process temporal context (3D-CNN, I3D) or models that analyze frequency artifacts (patch-based classifiers).

Implementation notes:
- Replace `SmallCNN` in `models.py` with a wrapper around torchvision models, e.g., `torchvision.models.resnet50(pretrained=True)` and change the final FC to output 2 classes.

### 5) Training recipe

A recommended baseline:
- Input size: 224x224 (if using torchvision pretrained models)
- Optimizer: AdamW or SGD with momentum
- Learning rate: 1e-4 to fine-tune pretrained networks (use warmup + cosine decay or step LR)
- Batch size: as large as GPU memory allows (16-64)
- Epochs: 10-50 depending on dataset size
- Loss: cross-entropy

Example (replace in `train.py`):
- Use a DataLoader built from `FFPPFramesDataset` with proper transforms
- Use a learning-rate scheduler

### 6) Adversarial training and evaluation

To detect adversarially attacked deepfakes, you want to evaluate both the detector's performance on clean fakes and on adversarial examples targeted to fool the detector.

Two approaches:
1. Adversarial evaluation (the prototype): craft adversarial examples using FGSM/PGD against the trained detector and measure accuracy.
2. Adversarial training: augment training with adversarial examples (e.g., PGD) to make the detector robust.

Adversarial training recipe (simple):
- For each training batch, craft adversarial images using current model weights (PGD with small eps), then compute loss on a mixture of clean and adversarial images or only adversarial images.
- Update weights on that loss.

Considerations:
- Computational cost: PGD is expensive. FGSM is cheaper but less robust.
- Eps scheduling: start with small epsilon and grow, or tune by validation.

### 7) Metrics and evaluation

Report:
- Accuracy on clean validation set
- Accuracy on adversarial validation set (FGSM, PGD with different epsilons)
- ROC AUC, precision, recall per-class

Consider cross-dataset evaluation: train on one dataset (FF++) and test on Celeb-DF to assess generalization.

### 8) Practical tips and caveats

- Data balance: many datasets may be imbalanced. Use weighted sampling or focal loss.
- Face alignment: consistent crops and alignment help performance. If using a pretrained backbone, match the preprocessing pipeline.
- Distributed training: for large datasets, use multi-GPU with `torch.distributed` or `torchrun`.
- Reproducibility: log seeds and configuration.

### 9) Example code snippets to swap in a real dataset and pretrained model

1) Replace dataset in `train.py`:
```python
from datasets.ffpp_dataset import FFPPFramesDataset
train_ds = FFPPFramesDataset('data/ffpp/train.csv', transform=train_transform)
val_ds = FFPPFramesDataset('data/ffpp/val.csv', transform=val_transform)
```

2) Replace model with ResNet50 backbone:
```python
import torchvision.models as models
backbone = models.resnet50(pretrained=True)
backbone.fc = torch.nn.Linear(backbone.fc.in_features, 2)
model = backbone.to(device)
```

3) Example adversarial training step (PGD, in the train loop):
```python
# create adversarial batch
adv_imgs = pgd_attack(model, imgs, labels, epsilon=eps, alpha=alpha, iters=iters)
# mix or replace
imgs_for_loss = torch.cat([imgs, adv_imgs], dim=0)
labels_for_loss = torch.cat([labels, labels], dim=0)
outputs = model(imgs_for_loss)
loss = F.cross_entropy(outputs, labels_for_loss)
loss.backward(); optimizer.step()
```

### 10) Logging, checkpoints, and scaling

- Save best model by validation metric.
- Use TensorBoard or Weights & Biases to track metrics, including adversarial curves.
- Checkpoint model + optimizer + scheduler.

## Verification and tests I ran

- Installed dependencies to a venv and executed `train.py --epochs 1 --batch-size 32 --smoke`.
- Fixed a runtime bug related to gradient usage for FGSM and label tensor creation.
- Confirmed training completes and adversarial eval runs.

## Where to go from here (concrete next tasks)

- Replace `SyntheticDeepfakeDataset` with an `FFPPFramesDataset` (I can implement this for you if you point me to where the FF++ frames are on disk or I can provide a downloader script).
- Add adversarial training in the training loop and compare model robustness.
- Swap in a pretrained ResNet or EfficientNet backbone for better accuracy.
- Add unit tests and continuous integration.

---

If you want, I can now implement one of these next steps end-to-end: e.g., create a `datasets/ffpp.py` loader and modify `train.py` to train on a directory of extracted frames. Tell me which dataset and whether frames are already extracted (and their path), and I will implement it.