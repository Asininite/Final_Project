# Carlini-Wagner (C&W) Attack — explanation and implementation notes

This document explains the Carlini & Wagner (C&W) attack, its intuition, and provides a PyTorch implementation outline and integration notes for this repository.

## High-level intuition

Carlini & Wagner (2017) proposed a family of powerful targeted attacks that minimize a combination of distortion and loss designed to push the model to misclassify. Instead of a simple gradient sign step, C&W formulates an optimization problem:

Minimize: ||\delta||_p + c * f(x + \delta)
Subject to: x + \delta \in [0,1]^n

Here f(·) is a function that measures how close the model is to making the desired misclassification (for targeted attacks) or to decrease confidence in the true class (for untargeted). The method performs optimization in a change-of-variables space using tanh to enforce box constraints, and uses Adam to solve the optimization.

C&W is typically very effective but computationally expensive per-example (requires many optimizer iterations). It often outperforms FGSM/PGD in creating minimal perturbation adversarial examples.

## C&W attack components

1. Change of variables: represent adversarial example as
   x' = 0.5 * (tanh(w) + 1) to enforce x' in [0,1], and solve for w.
2. Objective:
   minimize ||x' - x||_2^2 + c * loss_term(x')
   where loss_term pushes logits towards target class.
3. Binary search on c (constant balancing distortion and adversarial loss) to find minimal perturbation.
4. Use Adam for optimization over w.

## PyTorch outline (untargeted attack)

The following is a simple implementation outline suitable for integration in `attacks.py`. It's intentionally straightforward and not highly optimized.

```python
import torch
import torch.nn.functional as F

def cw_attack(model, images, labels, targeted=False, target_labels=None, c=1e-2, kappa=0, steps=1000, lr=1e-2):
    # images: tensor BxCxHxW in [0,1]
    # labels: true labels (if untargeted) or source labels (if targeted)
    # target_labels: required if targeted=True

    device = images.device
    batch_size = images.size(0)

    # change of variables
    # arctanh to map from [0,1] to real numbers for w initialization
    def arctanh(x):
        return 0.5 * torch.log((1+x) / (1-x))

    imgs_t = images * 2.0 - 1.0  # map to [-1,1]
    w = arctanh(imgs_t * 0.999999).detach()
    w = w.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(x_adv)
        if targeted:
            if target_labels is None:
                raise ValueError('target_labels required for targeted attack')
            # encourage target
            f = F.relu((logits * 1.0).max(1)[0] - logits.gather(1, target_labels.unsqueeze(1)).squeeze(1) + kappa)
        else:
            # encourage non-target (reduce correct class logit)
            correct_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other_max = (logits + (torch.eye(logits.size(1))[labels].to(device) * -1e4)).max(1)[0]
            f = F.relu(correct_logit - other_max + kappa)
        loss1 = ((x_adv - images) ** 2).view(batch_size, -1).sum(1)
        loss2 = c * f
        loss = (loss1 + loss2).sum()
        loss.backward()
        optimizer.step()
    x_adv = 0.5 * (torch.tanh(w) + 1.0).detach()
    return x_adv
```

Notes:
- This is a single-run implementation; standard C&W runs a binary search over `c` to find the smallest distortion that causes misclassification.
- The `kappa` parameter is a confidence margin; larger values produce stronger misclassifications but may increase distortion.
- The method is slow: `steps` may be several hundred to thousands.

## Integration notes

- Use C&W as a strong adversary for evaluation and for crafting robust training examples (though adversarial training with C&W is less common due to cost).
- If you need targeted attacks, pass `targeted=True` and `target_labels` of shape `(B,)` with desired classes.
- For batch processing, this approach runs optimization per-batch but still high cost. Consider running on a small validation subset for robustness evaluation.

## References
- Nicholas Carlini and David Wagner. "Towards Evaluating the Robustness of Neural Networks." 2017.

---

If you'd like, I can implement `cw_attack` in `attacks.py` now, add CLI options to `train.py` to evaluate with C&W on a small validation subset, and run a quick smoke test (on a few synthetic samples) to ensure the function runs. This will be slow if used on large datasets; I'd limit steps to 100 for a quick check.