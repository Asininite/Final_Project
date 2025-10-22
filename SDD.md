# Software Design Document (SDD)

Project: Adversarial Deepfake Detector (prototype)
Location: d:/FINAL_PROJECT
Date: 2025-10-14
Author: (auto-generated)

## 1. Overview
This SDD describes the architecture, components, data flow, algorithms, and testing strategy for the Adversarial Deepfake Detector prototype present in the workspace. It maps the functional requirements from the SRS to concrete modules and code files.

## 2. High-level architecture

Components and file mapping:
- Data preprocessing: `tools/face_preprocess.py`
- Datasets: `datasets.py` (synthetic), `datasets/ffpp_dataset.py` (real frames)
- Models: `models.py` (SmallCNN), optional backbones (to be added: ResNet50/MobileNetV3 wrappers)
- Attacks: `attacks.py` (FGSM, PGD), `carlini_wagner.md` contains C&W design; optional add `attacks.cw` implementation
- Training orchestration: `train.py`
- Utilities: `utils.py`
- Documentation: `README.md`, `IMPLEMENTATION_DETAILS.md`, `FUTURE_STEPS.md`, `teaching.md`, `carlini_wagner.md`

Deployment: local experiments and notebooks. No production server is included.

## 3. Module design

3.1 `tools/face_preprocess.py`
- Purpose: extract frames, run MTCNN for face detection, crop, resize, and write a `catalog.csv` with `path,label,video`.
- Inputs: directory of videos.
- Outputs: cropped images, `catalog.csv`.
- External deps: `ffmpeg` (system), `facenet-pytorch` (optional), `Pillow`.

3.2 `datasets/ffpp_dataset.py`
- Purpose: read `catalog.csv`, apply transforms (Resize/CenterCrop/ToTensor/Normalize), and return `(img_tensor, label)`.
- Data contract: expects file paths relative to CWD or absolute paths.
- Transform default: ImageNet normalization when using pretrained backbones.

3.3 `datasets.py` (SyntheticDeepfakeDataset)
- Purpose: generate quick synthetic images and optional adversarially attacked variants for smoke testing.
- Useful for unit tests and quick validation without large datasets.

3.4 `models.py` (SmallCNN)
- Small convolutional architecture with 3 conv layers, pooling and classifier.
- API: `forward(x)` returning logits shape `(B, num_classes)`.

3.4.1 Pretrained backbone wrappers
- Add a new module `backbones.py` (or extend `models.py`) containing wrappers for pretrained networks: ResNet50 and MobileNetV3 (small and large). Each wrapper will:
  - Load torchvision pretrained weights when requested.
  - Replace the final classifier layer with a 2-class output.
  - Offer an option to freeze layers except classifier (useful for transfer learning).

Example API:
```python
from backbones import get_backbone
model = get_backbone('mobilenet_v3_large', num_classes=2, pretrained=True, freeze_backbone=False)
```

3.5 `attacks.py`
- FGSM: `fgsm_attack(model, images, labels, epsilon)` — one-step sign method.
- PGD: `pgd_attack(model, images, labels, epsilon, alpha, iters)` — iterative projected gradient method.
- C&W: design in `carlini_wagner.md`; recommended to implement as `cw_attack(model, images, ...)` when needed.

3.5.1 Adversarial training support
- Add an `adv_train` option in `train.py` that accepts values `none`, `fgsm`, `pgd`.
- When `adv_train == 'pgd'`, before each optimizer step generate adversarial examples using `pgd_attack` with the specified `eps`, `alpha`, and `iters` and compute loss on these adversarial inputs. Allow mixing ratios between clean and adversarial examples (configurable).

Pseudo-logic:
```
if adv_train == 'pgd':
  adv_imgs = pgd_attack(model, imgs, labels, eps, alpha, iters)
  imgs_for_loss = torch.cat([imgs, adv_imgs], dim=0)
  labels_for_loss = torch.cat([labels, labels], dim=0)
  outputs = model(imgs_for_loss)
  loss = cross_entropy(outputs, labels_for_loss)
```

3.6 `train.py`
- Responsibilities:
  - Parse CLI args
  - Build datasets and dataloaders (synthetic or real)
  - Instantiate model and optimizer
  - Run training epochs (calls to `train_epoch`) and validation (`eval_epoch`)
  - Run adversarial evaluation using attack functions
- Extensibility points:
  - Add CLI flags for dataset path, backbone choice, adv-training mode, checkpoint path, logging backend.

Additional CLI and config requirements to implement:
- `--backbone {smallcnn,resnet50,mobilenet_v3_small,mobilenet_v3_large}`
- `--pretrained` (bool) to enable pretrained ImageNet weights
- `--freeze-backbone` (bool)
- `--adv-train {none,fgsm,pgd}` with `--eps`, `--pgd-steps`, `--pgd-alpha`
- `--save-dir PATH` to write checkpoints
- `--resume PATH` to resume training from checkpoint

Checkpoint format:
- Save `{'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'epoch': epoch, 'rng_states': {...}}` as `pth` using `torch.save`.

Resume logic:
- On `--resume path`, load checkpoint and restore model/optimizer/scheduler states and RNG states. Continue training from `epoch+1`.

3.7 `utils.py`
- Helper functions: `count_parameters`, `to_device` (moves batch to device and ensures label tensor types).

## 4. Data flow
1. Preprocessing: videos -> frames (ffmpeg) -> face crops (MTCNN) -> stored images + `catalog.csv`.
2. `FFPPFramesDataset` reads `catalog.csv`, yields tensors and labels.
3. `train.py` loads DataLoader batches and sends to model on device.
4. Training: forward -> loss -> backward -> optimizer step.
5. Adversarial evaluation: `eval_epoch` optionally crafts adversarial examples using provided attack function and evaluates model.

## 5. Data formats and shapes
- Images: float32 tensors, shape `(B, 3, H, W)` (H and W typically 224 for pretrained backbones, 64 for SmallCNN prototype).
- Labels: `torch.LongTensor` shape `(B,)` with values 0 or 1.
- Catalog CSV: `path,label,video` with `path` relative to repository root or absolute.

## 6. Interfaces and APIs
- `fgsm_attack(model, images, labels, epsilon) -> adv_images`
- `pgd_attack(model, images, labels, epsilon, alpha, iters) -> adv_images`
- `FFPPFramesDataset(catalog_csv, transform) -> Dataset` returns `(img, label)`.
- `train.py` CLI: `--epochs`, `--batch-size`, `--smoke` (extendable)

## 7. Algorithms and pseudo-code

7.1 Training loop (high-level):
```
for epoch in range(epochs):
    for imgs, labels in train_loader:
        imgs, labels = to_device((imgs, labels), device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    validate()
```

7.2 FGSM attack (one-step):
```
images.requires_grad = True
logits = model(images)
loss = cross_entropy(logits, labels)
loss.backward()
adv = clamp(images + epsilon * sign(images.grad), 0,1)
```

7.3 PGD attack (iterative):
```
pert = images.clone()
for i in range(iters):
    pert.requires_grad = True
    logits = model(pert)
    loss = cross_entropy(logits, labels)
    loss.backward()
    pert = pert + alpha * sign(pert.grad)
    pert = project(pert, images, epsilon)
    pert = clamp(pert,0,1)
```

## 8. Configuration and hyperparameters
Default hyperparameters (in repository):
- lr = 1e-3 (Adam for SmallCNN)
- batch_size = 64 (reduced in `--smoke` mode)
- epochs = 5 (default)
- FGSM eps = 0.03
- PGD eps = 0.03, alpha = 0.01, iters = 10

These should be exposed via CLI or config file for experiments.

## 9. Testing strategy
Unit tests and checks:
- Import tests for each module (datasets, models, attacks) to ensure no syntax errors.
- Functional smoke test: `python train.py --epochs 1 --batch-size 32 --smoke` must complete and report metrics.
- Preprocessing test: run `tools/face_preprocess.py` on a short sample video to validate `catalog.csv` generation.
- Attack correctness: test FGSM reduces accuracy of a trained model on a small validation set.

## 10. Performance and scaling
- For large datasets and adversarial training (PGD), use multiple GPUs and mixed precision.
- Use `torch.distributed` with `DistributedSampler` to scale across nodes.
- Consider LMDB or binary caches for fast image I/O.

## 11. Security and privacy
- Do not commit raw face images or private datasets to the repository.
- Warn users to comply with dataset licenses and privacy policies.

## 12. CI/CD and automation
- Provide a lightweight test matrix in CI that runs import and smoke tests on CPU or a small GPU runner.
- Add GitHub Actions or similar to run unit/import tests on PRs.

## 13. Deployment and reproducibility
- Save checkpoints and logs; include a `run.sh` or `run.ps1` script for reproducing experiments.
- Pin dependency versions in `requirements.txt`.

## 14. Future enhancements
- Implement backbone wrappers (ResNet50/MobileNetV3) and config-driven selection.
- Add adversarial training options (PGD/TRADES) and logging for robust metrics.
- Implement C&W attack and integrate for evaluation.
- Add visualization tools for adversarial perturbations and decision boundaries.

## 15. Appendix: file map
- `train.py` — training & eval
- `datasets.py` — synthetic dataset
- `datasets/ffpp_dataset.py` — real frames dataset loader
- `models.py` — SmallCNN
- `attacks.py` — FGSM and PGD
- `tools/face_preprocess.py` — frame extraction & face cropping
- `utils.py` — helpers
- docs: `README.md`, `IMPLEMENTATION_DETAILS.md`, `FUTURE_STEPS.md`, `teaching.md`, `carlini_wagner.md`

(End of SDD)
