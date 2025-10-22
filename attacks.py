import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon=0.03):
    """Simple FGSM attack that returns perturbed images in [0,1]."""
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    sign_data_grad = images.grad.sign()
    perturbed = images + epsilon * sign_data_grad
    return torch.clamp(perturbed, 0, 1)


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=10):
    images_orig = images.clone().detach()
    perturbed = images.clone().detach()
    perturbed.requires_grad = True
    for i in range(iters):
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        if perturbed.grad is not None:
            perturbed.grad.data.zero_()
        loss.backward()
        grad = perturbed.grad.data.sign()
        perturbed = perturbed + alpha * grad
        eta = torch.clamp(perturbed - images_orig, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images_orig + eta, 0, 1).detach()
        perturbed.requires_grad = True
    return perturbed


def cw_attack(model, images, labels, targeted=False, target_labels=None, c=1e-2, kappa=0, steps=200, lr=1e-2):
    """Carlini-Wagner L2 attack (basic implementation).

    Args:
        model: classifier model
        images: tensor BxCxHxW in [0,1]
        labels: true labels (untargeted) or source labels (targeted)
        targeted: whether to run targeted attack
        target_labels: required if targeted=True
        c: trade-off constant
        kappa: confidence margin
        steps: optimizer steps
        lr: learning rate for Adam
    Returns:
        adversarial images in [0,1]
    """
    device = images.device
    batch_size = images.size(0)

    def arctanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    imgs_t = images * 2.0 - 1.0
    # small scaling to keep inside open interval (-1,1)
    w = arctanh(imgs_t.clamp(-0.999999, 0.999999)).detach()
    w = w.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(x_adv)
        if targeted:
            if target_labels is None:
                raise ValueError('target_labels required for targeted attack')
            target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
            other_max = (logits + torch.eye(logits.size(1), device=device)[target_labels] * -1e4).max(1)[0]
            f = F.relu(other_max - target_logits + kappa)
        else:
            correct_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            # mask out correct class to get max of others
            mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
            other_max = (logits - mask * 1e4).max(1)[0]
            f = F.relu(correct_logit - other_max + kappa)
        l2 = ((x_adv - images) ** 2).view(batch_size, -1).sum(1)
        loss = (l2 + c * f).sum()
        loss.backward()
        optimizer.step()

    x_adv = 0.5 * (torch.tanh(w) + 1.0).detach()
    return x_adv.clamp(0, 1)
