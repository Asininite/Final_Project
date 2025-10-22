# Software Requirements Specification (SRS)

Project: Adversarial Deepfake Detector (prototype)
Location: d:/FINAL_PROJECT
Date: 2025-10-14
Author: (auto-generated)

## 1. Purpose
This document specifies the functional and non-functional requirements for the Adversarial Deepfake Detector prototype. The system detects whether an input face image/frame is a deepfake and evaluates robustness to adversarial perturbations (FGSM, PGD, C&W). The SRS is intended for developers, testers, and researchers who will extend or use the project.

## 2. Scope
The prototype is a research/engineering codebase intended for:
- Rapid experimentation with detectors for deepfakes and adversarial attacks.
- Smoke testing and reproducible small-scale experiments using synthetic and frame-level datasets.

It is not a production-grade detector (no hardened deployment, no privacy guarantees). The project contains preprocessing tools, dataset loaders, small and pretrained model options, attack implementations, and training/evaluation scripts.

## 3. Stakeholders
- Researchers developing and evaluating adversarial robustness for deepfake detection.
- Engineers who will integrate real datasets (FF++, Celeb-DF, DFDC).
- Students learning adversarial ML techniques.

## 4. Definitions and Acronyms
- FF++: FaceForensics++ dataset
- FGSM: Fast Gradient Sign Method
- PGD: Projected Gradient Descent
- C&W: Carlini-Wagner attack
- SRS: Software Requirements Specification
- SDD: Software Design Document

## 5. System Overview
The system has these logical components:
- Preprocessing: frame extraction and face cropping (`tools/face_preprocess.py`).
- Datasets: synthetic dataset (`datasets.py`) and real-frame dataset loader (`datasets/ffpp_dataset.py`).
- Models: `SmallCNN` (lightweight), option to swap pretrained backbones (ResNet50/MobileNetV3).
- Attacks: FGSM, PGD, (C&W documented in `carlini_wagner.md`).
- Training and evaluation: `train.py` orchestrates training, validation, and robustness evaluation.
- Utilities and docs: helpers and markdown instructions.

## 6. Functional Requirements (FR)
FR-1: Data ingestion
- The system shall accept pre-extracted frames or crops via a CSV catalog containing rows (path,label,video).
- The system shall accept synthetic data (for smoke tests) without external dependencies.

FR-2: Preprocessing
- The system shall extract frames from videos using `ffmpeg` and detect faces using MTCNN (if available) and save cropped images with labels.

FR-3: Model training
- The system shall train a model (SmallCNN or chosen backbone) on labeled images using cross-entropy loss.
- The system shall support configurable hyperparameters: learning rate, batch size, epochs, and optimizer.

FR-4: Adversarial attack generation
- The system shall generate adversarial examples using FGSM and PGD and optionally C&W for evaluation.

FR-5: Adversarial evaluation
- The system shall evaluate model performance on clean and adversarially perturbed validation sets and report loss and accuracy.

FR-6: Extensibility
- The system shall allow swapping the backbone model to pretrained ResNet50 or MobileNetV3 and toggling freezing/unfreezing layers.

FR-8: Checkpointing and resume
- The system shall save model checkpoints (model state_dict, optimizer state, scheduler state, epoch, RNG seeds) to disk at configurable intervals and when validation improves.
- The system shall provide a mechanism to resume training from a checkpoint and continue training/evaluation.

FR-9: Adversarial training
- The system shall support adversarial training using PGD (configurable eps, alpha, iterations) as a training mode option. The mode shall be selectable via CLI/config and expose hyperparameters.

FR-10: Backbone selection
- The system shall support selecting backbone models (e.g., `smallcnn`, `resnet50`, `mobilenet_v3_large`, `mobilenet_v3_small`) via CLI flags or a config file and automatically apply the appropriate preprocessing transforms (ImageNet normalization for pretrained backbones).

FR-7: CLI and configuration
- The system shall provide `train.py` with CLI flags for epochs, batch-size, smoke-mode. It shall be straightforward to add more flags (dataset path, backbone, adv-training options).

## 7. Non-functional Requirements (NFR)
NFR-1: Usability
- Developers should be able to run a smoke test in under a few minutes on a modern laptop.

NFR-2: Reproducibility
- Experiments shall document important hyperparameters; code should allow quick smoke tests to confirm reproducibility.

NFR-3: Performance
- The prototype should be efficient enough to run small experiments; full-scale training times are expected to be high and depend on hardware.

NFR-4: Security and privacy
- The prototype will not collect or exfiltrate user data. When working with face datasets, users must follow dataset licensing and privacy rules off-repository.

NFR-5: Maintainability
- Code should be modular (datasets, models, attacks) to ease extension.

## 8. Data Requirements
- Image tensors: shape (C,H,W), float32, normalized to [0,1] or ImageNet mean/std if using pretrained models.
- Catalog CSV schema: columns `path,label,video` where `label` ∈ {0,1}.
- Model checkpoint: PyTorch `state_dict` (binary).

## 9. Interfaces
- Command-line interface: `train.py` parameters.
- Filesystem: dataset csv and image files; output directories for checkpoints and logs.

## 10. Acceptance Criteria
- A developer can run `python train.py --epochs 1 --batch-size 32 --smoke` and observe training and evaluation output without runtime errors.
- Dataset loader `datasets/ffpp_dataset.py` must load `catalog.csv` and return `(img_tensor, label)` pairs.
- `tools/face_preprocess.py` must extract frames and create `catalog.csv` for small test videos.
- FGSM and PGD attacks must produce perturbed images with a measurable drop in detection accuracy on a trained model.
 - The training script shall be able to save and load checkpoints. Example: `--save-dir` argument saves best and last checkpoints; `--resume /path/to/checkpoint.pth` resumes training.
 - The training script shall be able to run adversarial training: e.g. `--adv-train pgd --eps 8/255 --pgd-steps 7` successfully completes and produces a model file.
 - The training script shall accept a `--backbone mobilenet_v3_large` option that constructs a MobileNetV3-based model and applies ImageNet preprocessing.

## 11. Constraints and Assumptions
- Assumes `ffmpeg` is available on PATH for preprocessing.
- Assumes PyTorch and torchvision are installed in the environment.
- The prototype is not intended for adversarial robustness certification.

## 12. Risks
- Running attacks and adversarial training can be computationally expensive.
- Using real face datasets may impose licensing and privacy constraints.

## 13. Appendix (commands)
- Run smoke test:
```powershell
python train.py --epochs 1 --batch-size 32 --smoke
```
- Extract frames and crops:
```powershell
python tools/face_preprocess.py --video-dir path/to/videos --out-dir data/ffpp/aligned --fps 1
```


----

(End of SRS)
