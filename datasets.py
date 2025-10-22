import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDeepfakeDataset(Dataset):
    """Generates synthetic images for prototype.

    Each sample is a 3x64x64 tensor. Labels:
      0 = real
      1 = fake (either synthesized or adversarially attacked)
    """
    def __init__(self, length=1000, image_size=64, attack_fn=None, attack_prob=0.5):
        self.length = length
        self.image_size = image_size
        self.attack_fn = attack_fn
        self.attack_prob = attack_prob

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Create a random "real" image pattern
        img = np.random.rand(3, self.image_size, self.image_size).astype(np.float32)
        label = 0

        # Randomly decide if this sample is a "fake" (synthesized) or adversarially attacked
        if np.random.rand() < 0.5:
            # synthesized fake: add a smooth blob
            x = np.linspace(-1, 1, self.image_size)
            xv, yv = np.meshgrid(x, x)
            blob = np.exp(-((xv**2 + yv**2) / 0.2))[None, :, :]
            img = img * 0.5 + 0.5 * blob.astype(np.float32)
            label = 1
        elif self.attack_fn is not None and np.random.rand() < self.attack_prob:
            # adversarial attack applied to a "fake" sample
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            with torch.no_grad():
                adv = self.attack_fn(img_tensor)
            img = adv.squeeze(0).numpy()
            label = 1

        return torch.from_numpy(img), label
