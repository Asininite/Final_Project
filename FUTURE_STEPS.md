# Future steps to finish the Adversarial Deepfake Detector project

This file lists detailed, actionable steps to complete the project using a real deepfake dataset (e.g., FaceForensics++, Celeb-DF, DFDC). It also explains how to implement adversarial training and how to swap the prototype's `SmallCNN` for a pretrained ResNet50 or MobileNetV3.

Contents:
- Dataset acquisition and storage
- Frame extraction and face preprocessing
- Implementing a dataset loader and data splits
- Model backbones: swapping to ResNet50 and MobileNetV3 (code snippets)
- Adversarial training: design, code example, and hyperparameters
- Evaluation: metrics and protocols
- Scaling, checkpoints, and recommended tooling

---

## 1) Dataset acquisition and storage

1. Choose dataset(s): FF++ (FaceForensics++), Celeb-DF, DFDC.
2. Download according to the dataset's instructions. Store under a top-level `data/` directory via:

```
data/
  ffpp/
    videos/
    frames/        # extracted frames here
    aligned/       # optional face-cropped aligned images
    labels.csv     # mapping frame -> label
```

3. Use a CSV or directory structure that maps each image path to a label (0 real, 1 fake). Example CSV header:
```
path,label,source_video
frames/real/video1_00001.png,0,video1
frames/fake/video2_00001.png,1,video2
```

Notes:
- Keep metadata for provenance (which manipulation method, source, compression level) to allow targeted analyses.
- Dataset size: FF++ can be large; use a subset for early experiments (e.g., small number of videos) and scale up.

## 2) Frame extraction and face preprocessing

Questions to decide:
- Do you want frame-level detector (single-frame) or video-level (temporal)?

Recommended pipeline for frame-level detectors:
1. Extract frames using `ffmpeg` at 1-5 fps:
   ```powershell
   ffmpeg -i input.mp4 -vf fps=1 outdir/frame_%06d.png
   ```
2. Run face detection per frame (MTCNN, RetinaFace, or dlib detector). Save bounding boxes.
3. Crop and optionally align faces. Resize to the model input size (e.g., 224x224).
4. Save cropped images to `data/ffpp/aligned/<video_name>/frame_000123.png` and record in CSV.

Notes on face detection:
- MTCNN is easy to use (pip `facenet-pytorch`), RetinaFace is more accurate but heavier.
- For multiple faces in a frame, either pick the largest bounding box or treat each crop as a separate sample (with same label).

Face cropping script structure (pseudo):
```python
from facenet_pytorch import MTCNN
from PIL import Image
mtcnn = MTCNN(keep_all=True)
for frame_path in frames:
    img = Image.open(frame_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    if boxes is None: continue
    for i, box in enumerate(boxes):
        crop = img.crop(box)
        crop = crop.resize((224,224))
        crop.save(outpath)
        write_csv(outpath, label)
```

## 3) Implement Dataset loader

Create `datasets/ffpp_dataset.py` with a PyTorch `Dataset`:
- In `__init__`, load CSV into memory, set transforms for train/val/test.
- `__getitem__` loads the image via PIL, applies transforms, returns `(tensor, label)`.

Example transforms for pretrained backbones:
```python
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

Add a `get_dataloaders` helper that returns train/val/test `DataLoader`s with appropriate `batch_size`, `num_workers`, and `sampler` settings.

## 4) Swapping to pretrained ResNet50 or MobileNetV3

Why use pretrained backbones?
- They provide strong feature extractors trained on ImageNet, improving sample efficiency and final performance.

Where to change:
- Replace `SmallCNN` in `models.py` or add new file `backbones.py` containing wrappers.

ResNet50 code snippet:
```python
import torchvision.models as models
import torch.nn as nn

def resnet50_backbone(num_classes=2, pretrained=True, freeze_backbone=False):
    model = models.resnet50(pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

MobileNetV3 (small or large) snippet:
```python
def mobilenetv3_backbone(num_classes=2, pretrained=True, freeze_backbone=False, variant='large'):
    if variant == 'large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
    else:
        model = models.mobilenet_v3_small(pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # replace classifier
    in_features = model.classifier[0].in_features if hasattr(model, 'classifier') else None
    model.classifier = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return model
```

Notes:
- Pretrained weights require torchvision >= 0.13 (match with PyTorch version). If image normalization differs, ensure transforms match.
- Freezing the backbone for several epochs and then unfreezing can help stability (two-stage fine-tuning).

## 5) Adversarial training (detailed)

Goal: make the detector robust to adversarial perturbations. High-level strategies:
1. Minimax adversarial training: for each batch, generate adversarial examples (PGD) and train to minimize loss on those examples (and optionally on clean images too).
2. Mix clean and adversarial images in the batch and train on combined loss.

Choices and trade-offs:
- FGSM-based training is faster but less robust than PGD-based training.
- PGD with 8-10 steps yields stronger robustness but increases computation by ~8-10x.
- Use smaller eps when training on images scaled in [0,1]. Typical eps ranges for 224-sized images: 2/255 to 8/255.

Adversarial training code example (PGD integrated into training loop):
```python
# inside training loop per batch
model.train()
imgs, labels = imgs.to(device), labels.to(device)
# generate adversarial examples using current model
adv_imgs = pgd_attack(model, imgs, labels, epsilon=8/255, alpha=2/255, iters=7)
# combine or replace
imgs_combined = torch.cat([imgs, adv_imgs], dim=0)
labels_combined = torch.cat([labels, labels], dim=0)
outputs = model(imgs_combined)
loss = F.cross_entropy(outputs, labels_combined)
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Advanced variants:
- TRADES: trade-off between natural accuracy and robustness by adding Kullback-Leibler regularization between model outputs on clean and adversarial inputs.
- MART: focuses more on misclassified examples to improve robustness.

Hyperparameters to tune:
- eps (perturbation magnitude): typical values 2/255, 4/255, 8/255
- PGD steps and alpha (step size)
- ratio of adversarial to clean images in each batch
- optimizer and lr schedule

Memory and speed considerations:
- Use gradient accumulation if batch doubling (clean + adv) exceeds GPU memory.
- Mixed precision (AMP) reduces memory and speeds up training.

## 6) Evaluation protocol

- Evaluate on clean test set.
- Evaluate on adversarial test sets crafted using FGSM and PGD with multiple epsilons.
- Report standard metrics: accuracy, precision, recall, F1, and AUC.
- For robustness, plot accuracy vs epsilon curve.
- Cross-dataset test: train on FF++ and evaluate on Celeb-DF frames to quantify generalization.

## 7) Checkpoints, logging, and reproducibility

- Save checkpoints containing epoch, model state_dict, optimizer state_dict, scheduler state if any, and RNG seeds.
- Log metrics and images to TensorBoard or Weights & Biases.
- Save sample adversarial images with predicted labels for qualitative inspection.

## 8) Scaling up

- For multi-GPU, use `torch.distributed` with `torchrun` and `DistributedSampler`.
- Use faster dataset storage (LMDB) or a cached dataset class for quick read.
- Use `torch.backends.cudnn.benchmark = True` when input sizes are fixed.

## 9) Example `train.py` flags to add

Add CLI flags to `train.py`:
```
--dataset PATH
--backbone {smallcnn,resnet50,mobilenetv3}
--pretrained
--freeze-backbone
--adversarial-training {none,fgsm,pgd}
--eps FLOAT
--pgd-steps INT
--amp
--save-dir PATH
```

## 10) Timeline / checklist to complete project

1. Implement face extraction & cropping scripts (ffmpeg + MTCNN).
2. Implement `FFPPFramesDataset` with CSV cataloging.
3. Add pretrained backbone wrappers (ResNet50, MobileNetV3).
4. Add CLI options & config support (e.g., YAML or JSON config files).
5. Implement adversarial training option (PGD) and safe defaults.
6. Add checkpointing, resume, logging (TensorBoard/W&B).
7. Run baseline experiments on a small subset to validate pipeline.
8. Scale to full dataset and run final experiments.

---

If you want, I can now implement one specific part from the checklist: face extraction scripts, the `FFPPFramesDataset` class, or adding the ResNet50/MobileNetV3 wrappers and hooking them into `train.py`. Tell me which step you'd like me to implement next.