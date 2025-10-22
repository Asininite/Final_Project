import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FFPPFramesDataset(Dataset):
    """Dataset for pre-extracted and pre-cropped frames saved by `tools/face_preprocess.py`.

    Expects a CSV with columns: path,label,video
    The path should be relative to the project root or absolute.
    """
    def __init__(self, catalog_csv, transform=None):
        self.catalog_csv = Path(catalog_csv)
        assert self.catalog_csv.exists(), f"Catalog not found: {catalog_csv}"
        self.entries = []
        with open(self.catalog_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                path = Path(r['path'])
                label = int(r['label']) if r['label'] not in ['', 'None'] else -1
                self.entries.append((path, label))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        # allow relative paths
        if not path.is_absolute():
            path = Path.cwd() / path
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label
