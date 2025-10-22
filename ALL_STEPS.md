# All steps performed and future steps

This document summarizes everything completed so far in the project and the recommended future steps to finish and scale the adversarial deepfake detector.

## Summary of actions performed (done)

1. Project skeleton
   - Created core files: `train.py`, `datasets.py`, `models.py`, `attacks.py`, `utils.py`, `README.md`, `requirements.txt`.

2. Synthetic dataset
   - Implemented `SyntheticDeepfakeDataset` in `datasets.py` to produce synthetic "real" and "fake" images for fast prototyping.

3. Small CNN detector
   - Implemented `SmallCNN` in `models.py` for quick training and testing.

4. Attack implementations
   - Implemented FGSM and PGD attacks in `attacks.py`.

5. Training loop
   - Implemented `train.py` with `train_epoch`, `eval_epoch` and CLI flags. Supports smoke testing with `--smoke` to run quick experiments.

6. Utilities
   - Implemented `utils.py` helpers: `count_parameters` and `to_device`.

7. Smoke test and debugging
   - Installed dependencies in the workspace venv and ran `train.py --epochs 1 --batch-size 32 --smoke`.
   - Fixed a bug where adversarial attacks were attempted under `torch.no_grad()` and label tensor handling.

8. Documentation
   - Wrote `IMPLEMENTATION_DETAILS.md` and `FUTURE_STEPS.md` describing implementation, design decisions, and steps to adapt to real datasets.

9. Face preprocessing tool
   - Created `tools/face_preprocess.py` to extract frames with `ffmpeg`, detect faces with MTCNN (`facenet-pytorch`), crop and save aligned images, and create a `catalog.csv`.
   - Added `Pillow` and `facenet-pytorch` to `requirements.txt` and installed them.

10. FF++ dataset loader
    - Implemented `datasets/ffpp_dataset.py` that reads `catalog.csv` and returns image tensors and labels. Made `datasets` a package and validated import.

## Files added/modified
- `README.md` — quickstart
- `requirements.txt` — dependencies
- `train.py` — training/eval script
- `datasets.py` — synthetic dataset
- `datasets/ffpp_dataset.py` — FF++ frames dataset loader
- `models.py` — SmallCNN
- `attacks.py` — FGSM, PGD
- `utils.py` — helpers
- `tools/face_preprocess.py` — frame extraction and face cropping
- `IMPLEMENTATION_DETAILS.md`, `FUTURE_STEPS.md` — documentation
- `ALL_STEPS.md` — this file

## Future / recommended steps (high priority)

1. Implement and validate end-to-end on a small subset of a real dataset (FF++).
   - Extract a small number of frames (1 fps) and generate face crops using `tools/face_preprocess.py`.
   - Use `datasets/ffpp_dataset.py` to load data and run `train.py` with the dataset (add a `--dataset` flag to `train.py`).

2. Swap to a pretrained backbone (ResNet50 or MobileNetV3) for better performance. Fine-tune the final layers and optionally unfreeze progressively.

3. Implement adversarial training (PGD) inside `train.py` with CLI toggles and safe defaults.

4. Add checkpointing, logging (TensorBoard or W&B), and resume training capability.

5. Perform robust evaluation: clean accuracy, accuracy under FGSM/PGD at several epsilons, and cross-dataset testing (train on FF++, test on Celeb-DF).

6. Optimize for scale: DDP, mixed precision, data caching, and use of multiple workers.

## Future / recommended steps (research and experimentation)

- Experiment with temporal models (3D CNNs, I3D) to exploit temporal artifacts in videos.
- Test defenses like input preprocessing, ensemble models, and robust feature extractors.
- Compare adversarial training regimes (TRADES, MART) for a trade-off between natural accuracy and robustness.

## Next actionable item suggestions
1. Wire `datasets/ffpp_dataset.py` into `train.py` and add CLI options to choose dataset/backbone/adv-training.
2. Implement adversarial training in `train.py` and test on a small FF++ subset.
3. Replace `SmallCNN` with `resnet50_backbone` and re-run.

If you want, I can implement item 1 (wire dataset into `train.py`) now and run a short smoke test with a synthetic small catalog to verify end-to-end behavior.
