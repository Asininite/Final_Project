# Teaching guide: Building an Adversarial Deepfake Detector (from scratch)

This guide teaches you the whole process assuming you know basic Python programming but have little or no background in Machine Learning. It walks through concepts, practical steps, and the code in this repository.

Table of contents
- Part 1: Core concepts (ML basics, CNNs, adversarial attacks)
- Part 2: Project overview and files
- Part 3: Step-by-step setup and run
- Part 4: Understanding and modifying the code
- Part 5: Next steps and recommended learning path

---

## Part 1 — Core concepts (short and friendly)

1. What is a dataset?
   - A dataset is a collection of examples. For image tasks, each example is an image plus a label (e.g., real=0, fake=1).

2. What is a model?
   - A model is a function that takes an image and outputs a prediction. In deep learning, this function is parameterized by many numbers (weights) that we learn from data.

3. What is training?
   - Training is the process of adjusting the model's weights so its predictions match labels on training data. We use an optimizer like Adam or SGD and a loss function (cross-entropy for classification).

4. What is a CNN?
   - Convolutional Neural Networks (CNNs) are models designed for images. They use convolutional layers to extract visual features.

5. What are adversarial attacks?
   - Small perturbations added to images that change a model's prediction while being nearly imperceptible to humans. FGSM and PGD are common attacks.

6. What is adversarial training?
   - A defense where you include adversarial examples during training so the model learns to be robust to attacks.

---

## Part 2 — Project overview and files

- `train.py`: entrypoint to train and evaluate the detector.
- `datasets.py`: synthetic dataset to test code quickly.
- `datasets/ffpp_dataset.py`: dataset loader for real frames (loaded from CSV produced by preprocessing).
- `models.py`: small CNN detector.
- `attacks.py`: implementations of FGSM and PGD attacks.
- `tools/face_preprocess.py`: extracts frames and crops faces.
- `utils.py`: helper utilities.

---

## Part 3 — Step-by-step setup and run

1. Install Python (3.8+ recommended). Create a virtual environment (helps keep dependencies separate):
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run a smoke test (quick training on synthetic data):
```powershell
python train.py --epochs 1 --batch-size 32 --smoke
```
4. To prepare real data (FF++), extract frames and crop faces (see `tools/face_preprocess.py`):
```powershell
python tools/face_preprocess.py --video-dir path/to/videos --out-dir data/ffpp/aligned --fps 1
```
This writes cropped images and `catalog.csv` which you can load with `datasets/ffpp_dataset.py`.

---

## Part 4 — Understanding the code (walkthrough)

1. `datasets.SyntheticDeepfakeDataset`:
   - Creates random images and synthetic fakes. Useful for quick debugging when you don't want to download large datasets.

2. `models.SmallCNN`:
   - Small model with 3 conv layers and a classifier. Easy to understand and fast to run.

3. `attacks.fgsm_attack` and `attacks.pgd_attack`:
   - FGSM: one-step attack based on the sign of the gradient of loss w.r.t. the input.
   - PGD: iterative projected gradient descent attack; stronger than FGSM.

4. `train.py`:
   - `train_epoch` handles normal training.
   - `eval_epoch` supports evaluating on clean or adversarial examples (by passing `attack_fn`). Make sure the attack is generated with gradients enabled.

5. `tools/face_preprocess.py`:
   - Uses `ffmpeg` to extract frames and `facenet-pytorch` MTCNN to detect faces.

---

## Part 5 — Suggested learning path

1. Basic Python and NumPy.
2. PyTorch fundamentals (tensors, autograd, optim, DataLoader).
3. CNN basics and torchvision models.
4. Adversarial ML basics: FGSM, PGD, Carlini-Wagner.
5. Reading research papers and reproducing experiments.

---

If you'd like, I can walk you through any step interactively, or implement one of the next features (e.g., swapping to ResNet50 or adding adversarial training).

---

## Extended lessons — step-by-step teaching

The following sections teach the topics you requested in order. Each section includes: concise explanations, small code examples you can run, common pitfalls, and short exercises.

### 1) Basic Python and NumPy

Goal: be comfortable manipulating arrays, indexing, and basic plotting.

Key concepts:
- Python basics: variables, lists, dicts, loops, functions, context managers.
- NumPy: ndarrays, broadcasting, slicing, basic linear algebra operations.

Small examples:

1. NumPy array creation and slicing

```python
import numpy as np

# create a 3x4 array
a = np.arange(12).reshape(3,4)
print(a)

# slice first two rows, last two columns
print(a[:2, 2:])

# broadcasting example: add a vector to each row
v = np.array([1,2,3,4])
print(a + v)
```

2. Common pitfalls
- Mixing shapes: broadcasting rules are powerful but easy to misuse. When addition fails, print shapes.
- Using Python lists instead of NumPy arrays for numerical work is slower.

Exercise:
- Load an image with PIL, convert to NumPy array, compute mean and std per channel.

```python
from PIL import Image
img = Image.open('some_face.png').convert('RGB')
arr = np.array(img) / 255.0
means = arr.mean(axis=(0,1))
stds = arr.std(axis=(0,1))
print(means, stds)
```

### 2) PyTorch fundamentals (tensors, autograd, optim, DataLoader)

Goal: understand tensors, gradients, building simple models, and data pipelines.

Key concepts:
- torch.Tensor vs numpy.ndarray
- requires_grad and autograd
- nn.Module and parameter registration
- optimizers and training loop skeleton
- Dataset and DataLoader for batching

Small example: a minimal training loop (linear regression)

```python
import torch
from torch import nn

# create synthetic data y = 2x + 1
x = torch.randn(100,1)
y = 2*x + 1 + 0.1*torch.randn_like(x)

model = nn.Linear(1,1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(100):
   opt.zero_grad()
   pred = model(x)
   loss = loss_fn(pred, y)
   loss.backward()
   opt.step()
print('learned', model.weight.item(), model.bias.item())
```

DataLoader example (image classification style):

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DummyImageDataset(Dataset):
   def __init__(self, n=100):
      self.n = n
      self.transform = transforms.ToTensor()
   def __len__(self):
      return self.n
   def __getitem__(self, idx):
      import numpy as np
      img = (np.random.rand(3,64,64) * 255).astype('uint8')
      return self.transform(img), 0

loader = DataLoader(DummyImageDataset(), batch_size=16, shuffle=True)
for imgs, labels in loader:
   print(imgs.shape, labels.shape)
   break
```

Common pitfalls:
- Forgetting to call `optimizer.zero_grad()` — gradients accumulate.
- Not setting `model.train()` / `model.eval()` when appropriate.
- Mixing CPU/GPU tensors — pay attention to `.to(device)`.

Exercise:
- Modify the linear regression example to run on GPU (if available) and measure the time difference.

### 3) CNN basics and torchvision models

Goal: understand convolutions, pooling, and how to use pretrained models from torchvision.

Key concepts:
- Convolutional layer (Conv2d): kernels, stride, padding.
- Pooling (MaxPool, AvgPool) for downsampling.
- BatchNorm, Dropout
- Transfer learning with pretrained backbones (ResNet, MobileNet)

Simple CNN example (forward pass only):

```python
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv = nn.Sequential(
         nn.Conv2d(3, 16, 3, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(2),
         nn.Conv2d(16, 32, 3, padding=1),
         nn.ReLU(),
         nn.AdaptiveAvgPool2d((1,1)),
      )
      self.fc = nn.Linear(32, 2)
   def forward(self, x):
      x = self.conv(x)
      x = x.view(x.size(0), -1)
      return self.fc(x)

model = TinyCNN()
print(sum(p.numel() for p in model.parameters()))
```

Using pretrained ResNet50 from torchvision:

```python
import torchvision.models as models
resnet = models.resnet50(pretrained=True)
# replace final layer
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
```

Exercise:
- Load an example image using torchvision transforms and run through ResNet50 to get logits.

### 4) Adversarial ML basics: FGSM, PGD, Carlini-Wagner

Goal: understand attack goals, threat models, and how to implement FGSM/PGD quickly.

Key concepts:
- Threat model: what the attacker can change (pixel values), and the allowed perturbation (epsilon).
- Untargeted vs targeted attacks.
- White-box attacks (access to model & gradients) vs black-box attacks.

FGSM (Fast Gradient Sign Method) — code example:

```python
def fgsm_attack(model, images, labels, epsilon=0.03):
   images = images.clone().detach().requires_grad_(True)
   outputs = model(images)
   loss = nn.functional.cross_entropy(outputs, labels)
   loss.backward()
   pert = epsilon * images.grad.sign()
   adv = torch.clamp(images + pert, 0, 1)
   return adv
```

PGD (Projected Gradient Descent) — basic sketch:

```python
def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, iters=10):
   orig = images.clone().detach()
   pert = images.clone().detach()
   for i in range(iters):
      pert.requires_grad_(True)
      outputs = model(pert)
      loss = nn.functional.cross_entropy(outputs, labels)
      loss.backward()
      pert = pert + alpha * pert.grad.sign()
      pert = torch.min(torch.max(pert, orig - eps), orig + eps)
      pert = torch.clamp(pert.detach(), 0, 1)
   return pert
```

Carlini-Wagner: see `carlini_wagner.md` for math and code outline (it's optimization-based and slower but stronger).

Exercises:
- Implement FGSM and verify that model accuracy drops on adversarial examples.
- Try PGD with increasing epsilons and plot accuracy vs epsilon.

### 5) Reading research papers and reproducing experiments

Approach:
- Start with the abstract and conclusion to get the big picture.
- Reproduce a toy experiment first (use small datasets or fewer epochs).
- Pay attention to hyperparameters — many papers omit exact details; try to approximate and tune.

Practical tips:
- Keep configurations in a single place (YAML or JSON) for reproducibility.
- Run multiple seeds to test stability.
- Use logging (TensorBoard/W&B) and save checkpoints.

### 6) Swapping to ResNet50 (practical walkthrough)

Why swap?
- Pretrained ResNet50 provides better features and speeds up convergence.

Steps to swap in `train.py`:
1. Add an import: `import torchvision.models as models`
2. Replace model construction with:

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
```

3. Adjust transforms to use ImageNet normalization (already used in `FFPPFramesDataset`).
4. Use a smaller learning rate for pretrained weights (e.g., 1e-4) and consider freezing early layers:

```python
for name, param in model.named_parameters():
   if 'layer4' not in name and 'fc' not in name:
      param.requires_grad = False
```

5. Train for several epochs; unfreeze layers for fine-tuning if needed.

### 7) Adding adversarial training (practical walkthrough)

Goal: include adversarial examples during training to improve robustness.

Simple in-loop PGD adversarial training:

1. In your training loop, after loading a batch `(imgs, labels)`, create adversarial images using `pgd_attack(model, imgs, labels, eps, alpha, iters)`.
2. Option A: train only on adversarial images (standard adversarial training).
3. Option B: train on a mixture of clean and adversarial images (augmenting batch). This requires doubling the batch or using gradient accumulation.

Code sketch (inside `train_epoch`):

```python
adv_imgs = pgd_attack(model, imgs, labels, eps=8/255, alpha=2/255, iters=7)
imgs_comb = torch.cat([imgs, adv_imgs], dim=0)
labels_comb = torch.cat([labels, labels], dim=0)
outputs = model(imgs_comb)
loss = nn.functional.cross_entropy(outputs, labels_comb)
```

Practical tips:
- Use mixed precision (`torch.cuda.amp`) to reduce memory when doubling batches.
- Monitor both clean and adversarial validation accuracy.
- Tune eps carefully; large eps can harm clean accuracy.

### Final exercises and project ideas

- Implement adversarial training and compare clean vs robust accuracy.
- Swap in ResNet50 and compare convergence vs SmallCNN.
- Implement C&W and use it for a final robustness evaluation (on a small subset).

If you'd like, I can now implement either the ResNet50 swap in `train.py` or add adversarial training; tell me which one to do next and I'll implement it and run a short validation.

---

## Deep-dive: mathematical intuition, debugging, and resources

This section goes deeper into the math and intuition behind the main building blocks, gives troubleshooting tips, and lists high-quality resources for each topic so you can study further.

### A. Linear algebra and calculus refresher (why it matters)

Key ideas you should be comfortable with:
- Vectors and matrices: addition, multiplication, dot products.
- Matrix shapes and broadcasting — carefully track shapes when implementing layers.
- Gradients: derivative of scalar functions w.r.t. vectors; chain rule for composite functions.

Why it matters: neural networks compute a sequence of matrix/vector operations; training uses gradients (via backpropagation) to update parameters.

Resource primer:
- Linear Algebra: "Linear Algebra Done Right" (book) or Gilbert Strang's MIT OpenCourseWare lectures.
- Calculus: single-variable calculus and multivariable chain rule.

Small worked example: derivative of mean squared error for a single linear neuron

Given prediction y_hat = w*x + b and loss L = 0.5*(y_hat - y)^2, the gradients are:
- dL/dw = (y_hat - y) * x
- dL/db = (y_hat - y)

This is exactly what autograd computes for us in PyTorch.

### B. Convolutional math (intuitions you should hold)

Core intuition:
- Convolution applies a small kernel (filter) sliding over the image to compute local dot products. Each filter learns to detect a local pattern (edge, texture, etc.).
- Deeper layers combine earlier patterns into higher-level features.

Strides/padding:
- Stride controls how the filter moves; larger stride -> smaller output.
- Padding keeps spatial dimensions by adding border pixels.

Receptive field:
- As you add layers, each neuron's receptive field on the input grows — deeper neurons see larger context.

Resource primer:
- CS231n (Stanford) lecture notes are excellent for convolution intuition and visualizations.

### C. Backpropagation and autograd mechanics

High-level:
- Forward pass computes activations and caches intermediate values.
- Backward pass applies chain rule; gradients are propagated from outputs to inputs.

PyTorch specifics:
- Tensors with `requires_grad=True` track operations for autograd.
- `.backward()` computes gradients for all tensors involved.
- `.grad` on parameters stores gradients used by optimizers.

Debug tips:
- Use `torch.autograd.set_detect_anomaly(True)` to get helpful stack traces for NaNs or issues during backward.
- Print intermediate tensor stats (mean/std/min/max) to catch exploding/vanishing activations.

### D. Why adversarial examples exist (intuition)

Short intuition:
- Models are linear (locally) in high dimensions; small but well-chosen perturbations can move samples across decision boundaries.
- High-dimensional geometry makes it easy to find directions that change model output without perceptible change to humans.

Formal insights:
- Papers show adversarial vulnerability relates to model linearity and high-dimensional dot-product sensitivity.

Resource primer:
- Good starting papers: Szegedy et al. (2013), Goodfellow et al. (FGSM, 2014), Carlini & Wagner (2017). Also read Madry et al. (2017) for adversarial training.

### E. Carlini-Wagner deeper intuition

Why C&W is strong:
- It's an optimization-based attack that directly minimizes perturbation under a loss that targets model logits, and uses continuous optimization (Adam) with box constraints enforced via change-of-variable.

Practical note:
- C&W finds minimal-norm perturbations and is good for evaluation when you want to measure how small a perturbation is needed to cause misclassification.

### F. Debugging training and attacks

Common symptoms and fixes:
- Loss NaN or Inf: check data normalization, reduce learning rate, add gradient clipping, check for corrupted images.
- Gradients zero: ensure `requires_grad` is set on input when crafting attacks, and that there is a path from loss to inputs/parameters.
- Attack fails (model accuracy unchanged): confirm attack epsilon is large enough on normalized images (e.g., for images in [0,1], eps=8/255 is common).

Tools:
- TensorBoard, Weights & Biases for metrics.
- Save sample images before and after attack to visually inspect perturbations.

### G. Curated resources and learning path (ordered)

Beginner (Python & ML basics):
- "Python Crash Course" (book) or interactive Python tutorials on Codecademy.
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Chollet) — conceptual, not PyTorch-specific.

Core PyTorch learning:
- "Deep Learning with PyTorch" (book by Eli Stevens) — practical and thorough.
- Official PyTorch tutorials: https://pytorch.org/tutorials/
- fast.ai course (Practical deep learning) — top-down teaching with practical projects.

Computer Vision & CNNs:
- CS231n (Stanford) lecture notes and videos.
- "Deep Learning for Computer Vision" books and online resources.

Adversarial ML:
- "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2014)
- "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
- "Adversarial Examples Are Not Bugs, They Are Features" (Ilyas et al.)
- Madry et al. (2017) — adversarial training via PGD.

Advanced topics and codebases:
- CleverHans and Foolbox libraries for adversarial attacks.
- Model training at scale: PyTorch Distributed documentation and tutorials.

### H. Practical project workflow and reproducibility checklist

When running an experiment:
1. Record exact code version (git commit) and environment (python, torch versions).
2. Record dataset selection and preprocessing steps.
3. Save hyperparameters in a config file.
4. Run multiple seeds and aggregate results.
5. Save checkpoints and logs.

### I. Final recommended hands-on exercises (progressive)

1. Run the smoke test in this repo and inspect model outputs and training curves.
2. Implement FGSM and PGD attacks (already done here) and visualize perturbed images.
3. Swap in ResNet50 and fine-tune on a small dataset of faces.
4. Implement adversarial training (PGD) and report clean vs robust accuracy.
5. Implement or use C&W for a final robustness evaluation.

---

If you want, I can now run one of the following for you:
- Implement the ResNet50 swap automatically in `train.py` and run a 1-epoch smoke test.
- Add adversarial training to `train.py` and run a 1-epoch smoke test on synthetic data.
- Implement `cw_attack` in `attacks.py` and run a quick sanity check.

Tell me which and I will implement and run it now.