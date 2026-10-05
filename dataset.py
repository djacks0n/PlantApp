"""
Shared data code for normal_model.py and lookalike_model.py.

Both model files import from here, so they use the same class order,
the same transforms, and the same train/val/test split.

"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#invasive = even index & lookalike = odd index
PAIRS = [
    ("Ailanthus altissima", "Rhus copallinum"),          # 0, 1
    ("Elaeagnus umbellata", "Asimina triloba"),          # 2, 3
    ("Rubus phoenicolasius", "Rubus occidentalis"),      # 4, 5
    ("Lonicera japonica", "Campsis radicans"),           # 6, 7
    ("Hedera helix", "Parthenocissus quinquefolia"),     # 8, 9
    ("Persicaria perfoliata", "Persicaria sagittata"),   # 10, 11
    ("Alliaria petiolata", "Geum canadense"),            # 12, 13
    ("Ficaria verna", "Packera aurea"),                  # 14, 15
]

CLASS_NAMES = [name for pair in PAIRS for name in pair]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
print(CLASS_NAMES)
print(CLASS_TO_IDX)
NUM_CLASSES = len(CLASS_NAMES)

CONFUSABLE_PAIRS = [(2 * i, 2 * i + 1) for i in range(len(PAIRS))]
INVASIVE_IDX = set(range(0, NUM_CLASSES, 2))
SPLIT_DIR = Path("splits")

#transform for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class PlantDataset(Dataset):
    #Reads one of splits/train.csv, splits/val.csv, splits/test.csv

    def __init__(self, split_csv, transform):
        df = pd.read_csv(split_csv)
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def get_dataloaders(batch_size=32, num_workers=2):
    train_ds = PlantDataset(SPLIT_DIR / "train.csv", train_transform)
    val_ds = PlantDataset(SPLIT_DIR / "val.csv", eval_transform)
    test_ds = PlantDataset(SPLIT_DIR / "test.csv", eval_transform)

    GPU_available = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=GPU_available)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=GPU_available)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=GPU_available)

    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader