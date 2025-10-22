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
