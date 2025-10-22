import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import SyntheticDeepfakeDataset
from models import SmallCNN
from attacks import fgsm_attack
from utils import count_parameters, to_device


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs, labels = to_device((imgs, labels), device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, dataloader, device, attack_fn=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs, labels = to_device((imgs, labels), device)
        if attack_fn is not None:
            # generate adversarial examples targeting the detector (requires grads)
            imgs = attack_fn(model, imgs, labels)
            # After attack generation, we can evaluate without gradients
            with torch.no_grad():
                outputs = model(imgs)
        else:
            with torch.no_grad():
                outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # small dataset for smoke tests
    len_train = 256 if args.smoke else 2000
    len_val = 128 if args.smoke else 500

    train_ds = SyntheticDeepfakeDataset(length=len_train, attack_fn=None)
    val_ds = SyntheticDeepfakeDataset(length=len_val, attack_fn=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SmallCNN().to(device)
    print(f"Model params: {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device, attack_fn=None)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Now evaluate detector robustness: craft adversarial examples on validation set and measure
    adv_val = SyntheticDeepfakeDataset(length=256, attack_fn=None)
    adv_loader = DataLoader(adv_val, batch_size=args.batch_size, shuffle=False)
    # Use FGSM targeted against detector
    adv_loss, adv_acc = eval_epoch(model, adv_loader, device, attack_fn=fgsm_attack)
    print(f"Robust eval (FGSM)  loss={adv_loss:.4f}  acc={adv_acc:.4f}")

if __name__ == '__main__':
    main()
