import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import random

from datasets import SyntheticDeepfakeDataset
from datasets.ffpp_dataset import FFPPFramesDataset
from backbones import get_model_and_transforms
from attacks import fgsm_attack, pgd_attack, cw_attack
from utils import count_parameters, to_device
from losses import afsl_loss


def train_epoch(model, dataloader, optimizer, device,
                adv_train: str = 'none', eps: float = 0.03, pgd_steps: int = 7, pgd_alpha: float = 0.01, mix_ratio: float = 1.0,
                use_afsl: bool = False, afsl_attack: str = 'fgsm', lambda_adv: float = 1.0, lambda_sim: float = 1.0, sim_metric: str = 'cosine'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    # For AFSL component tracking
    afsl_sums = {'l_cls': 0.0, 'l_adv': 0.0, 'l_sim': 0.0}
    afsl_batches = 0
    for imgs, labels in dataloader:
        imgs, labels = to_device((imgs, labels), device)
        optimizer.zero_grad()
        if use_afsl:
            # AFSL combined loss over clean+adv with feature similarity
            loss, parts, logits_clean = afsl_loss(
                model, imgs, labels,
                attack=afsl_attack, eps=eps, pgd_steps=pgd_steps, pgd_alpha=pgd_alpha,
                lambda_adv=lambda_adv, lambda_sim=lambda_sim, sim_metric=sim_metric,
            )
            outputs = logits_clean  # for accuracy, use clean logits
            afsl_sums['l_cls'] += parts['l_cls']
            afsl_sums['l_adv'] += parts['l_adv']
            afsl_sums['l_sim'] += parts['l_sim']
            afsl_batches += 1
        else:
            if adv_train and adv_train != 'none':
                if adv_train == 'fgsm':
                    adv_imgs = fgsm_attack(model, imgs, labels, epsilon=eps)
                elif adv_train == 'pgd':
                    adv_imgs = pgd_attack(model, imgs, labels, epsilon=eps, alpha=pgd_alpha, iters=pgd_steps)
                else:
                    adv_imgs = None
                if adv_imgs is not None:
                    if mix_ratio >= 1.0:
                        imgs_for_loss = torch.cat([imgs, adv_imgs], dim=0)
                        labels_for_loss = torch.cat([labels, labels], dim=0)
                    else:
                        # mix a fraction of adv examples
                        b = imgs.size(0)
                        k = int(b * mix_ratio)
                        imgs_for_loss = torch.cat([imgs[:b-k], adv_imgs[:k]], dim=0)
                        labels_for_loss = torch.cat([labels[:b-k], labels[:k]], dim=0)
                else:
                    imgs_for_loss, labels_for_loss = imgs, labels
            else:
                imgs_for_loss, labels_for_loss = imgs, labels

            outputs = model(imgs_for_loss)
            loss = F.cross_entropy(outputs, labels_for_loss)
        loss.backward()
        optimizer.step()
        # Use batch size for aggregation; for AFSL, count clean batch size
        batch_count = imgs.size(0) if use_afsl else imgs_for_loss.size(0)
        total_loss += loss.item() * batch_count
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            target = labels if use_afsl else labels_for_loss
            correct += (preds == target).sum().item()
            total += batch_count
    metrics = None
    if use_afsl and afsl_batches > 0:
        metrics = {k: v / afsl_batches for k, v in afsl_sums.items()}
    return total_loss / total, correct / total, metrics


def eval_epoch(model, dataloader, device, attack_mode: str = 'none', eps: float = 0.03, pgd_steps: int = 10, pgd_alpha: float = 0.01):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs, labels = to_device((imgs, labels), device)
        if attack_mode and attack_mode != 'none':
            # generate adversarial examples targeting the detector (requires grads)
            if attack_mode == 'fgsm':
                imgs = fgsm_attack(model, imgs, labels, epsilon=eps)
            elif attack_mode == 'pgd':
                imgs = pgd_attack(model, imgs, labels, epsilon=eps, alpha=pgd_alpha, iters=pgd_steps)
            elif attack_mode == 'cw':
                imgs = cw_attack(model, imgs, labels, steps=min(200, pgd_steps * 20))
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
    parser.add_argument('--dataset-csv', type=str, default=None, help='Path to catalog.csv for FF++ frames')
    parser.add_argument('--backbone', type=str, default='smallcnn', help='smallcnn|resnet50|mobilenet_v3_small|mobilenet_v3_large')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')

    parser.add_argument('--adv-train', type=str, default='none', help='none|fgsm|pgd')
    parser.add_argument('--adv-eval', type=str, default='fgsm', help='none|fgsm|pgd|cw')
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--pgd-alpha', type=float, default=0.01)

    # AFSL options
    parser.add_argument('--use-afsl', action='store_true', help='Enable AFSL loss (CE + KL consistency + feature similarity)')
    parser.add_argument('--afsl-attack', type=str, default='fgsm', help='fgsm|pgd attack used to generate adversarial pairs for AFSL')
    parser.add_argument('--lambda-adv', type=float, default=1.0, help='Weight for AFSL adversarial consistency term')
    parser.add_argument('--lambda-sim', type=float, default=1.0, help='Weight for AFSL feature similarity term')
    parser.add_argument('--sim-metric', type=str, default='cosine', help='Similarity metric: cosine|mse')

    parser.add_argument('--save-dir', type=str, default='artifacts')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Model + transforms
    model, train_tf, val_tf = get_model_and_transforms(
        args.backbone, num_classes=2, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    print(f"Backbone: {args.backbone}  params: {count_parameters(model)}")

    # Datasets
    if args.dataset_csv:
        csv_path = Path(args.dataset_csv)
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        # Read CSV to compute video-based split
        rows = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                rows.append({**r, 'idx': i})
        videos = {}
        for r in rows:
            videos.setdefault(r['video'], []).append(r['idx'])
        vid_list = list(videos.keys())
        random.shuffle(vid_list)
        split = int(len(vid_list) * (0.8 if not args.smoke else 0.6))
        train_vids = set(vid_list[:split])
        val_vids = set(vid_list[split:])
        train_indices = [r['idx'] for r in rows if r['video'] in train_vids]
        val_indices = [r['idx'] for r in rows if r['video'] in val_vids]
        # In smoke mode, trim sizes
        if args.smoke:
            train_indices = train_indices[:512]
            val_indices = val_indices[:256]
        # Build datasets and wrap with subsets
        full_train = FFPPFramesDataset(csv_path, transform=train_tf)
        full_val = FFPPFramesDataset(csv_path, transform=val_tf)
        train_ds = Subset(full_train, train_indices)
        val_ds = Subset(full_val, val_indices)
    else:
        # small synthetic dataset for smoke tests or fallback
        len_train = 256 if args.smoke else 2000
        len_val = 128 if args.smoke else 500
        train_ds = SyntheticDeepfakeDataset(length=len_train, attack_fn=None)
        val_ds = SyntheticDeepfakeDataset(length=len_val, attack_fn=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state', ckpt))
            if 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            adv_train=args.adv_train, eps=args.eps, pgd_steps=args.pgd_steps, pgd_alpha=args.pgd_alpha,
            use_afsl=args.use_afsl, afsl_attack=args.afsl_attack, lambda_adv=args.lambda_adv, lambda_sim=args.lambda_sim, sim_metric=args.sim_metric,
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device, attack_mode='none')
        if args.use_afsl and train_metrics is not None:
            print(
                f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                f"AFSL(l_cls={train_metrics['l_cls']:.4f}, l_adv={train_metrics['l_adv']:.4f}, l_sim={train_metrics['l_sim']:.4f})"
            )
        else:
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # checkpointing
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': max(best_val_acc, val_acc),
        }
        torch.save(ckpt, save_dir / 'last.pt')
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, save_dir / 'best.pt')

    # Robustness evaluation on validation split
    adv_mode = args.adv_eval
    adv_loss, adv_acc = eval_epoch(
        model, val_loader, device, attack_mode=adv_mode, eps=args.eps, pgd_steps=args.pgd_steps, pgd_alpha=args.pgd_alpha
    )
    print(f"Robust eval ({adv_mode.upper() if adv_mode else 'NONE'})  loss={adv_loss:.4f}  acc={adv_acc:.4f}")

if __name__ == '__main__':
    main()
