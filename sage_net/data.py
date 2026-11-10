from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .prompts import normalize_class_name

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def scan_image_folder(root: str | Path) -> dict[str, list[str]]:
    """Read a dataset stored as root/<class>/<images>. Class names are normalised (e.g. 'DenseResidential' -> 'dense_residential')
    and the image lists are sorted, so sampling with a fixed seed is reproducible across machines."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
        subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not any(f.suffix.lower() in IMG_EXTENSIONS for f in subdirs[0].iterdir()):
        root = subdirs[0]

    classes: dict[str, list[str]] = {}
    for folder in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(str(f) for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS)
        if files:
            classes.setdefault(normalize_class_name(folder.name), []).extend(files)
    if not classes:
        raise RuntimeError(f"no images found under {root}")
    return classes


def build_transform(train: bool, size: int = 224):
    if train:
        ops = [
            T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    else:
        ops = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ImageList(Dataset):
    """(path, label) pairs. With cache=True the decoded images stay in memory,
    which helps for the small support sets that are revisited every epoch."""

    def __init__(self, items: Sequence[tuple[str, int]], transform, cache: bool = False):
        self.items = list(items)
        self.transform = transform
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if self.cache is None:
            img = load_rgb(path)
        else:
            if idx not in self.cache:
                self.cache[idx] = load_rgb(path)
            img = self.cache[idx]
        return self.transform(img), label


class EpisodeSampler:

    def __init__(self, classes: dict[str, list[str]], ways: int = 5, shots: int = 1,
                 queries: int = 15, seed: int = 0):
        self.classes = {c: v for c, v in classes.items() if len(v) >= shots + queries}
        if len(self.classes) < ways:
            raise ValueError(f"only {len(self.classes)} classes have {shots + queries} images")
        self.names = sorted(self.classes)
        self.ways, self.shots, self.queries = ways, shots, queries
        self.rng = random.Random(seed)

    def sample(self):
        chosen = sorted(self.rng.sample(self.names, self.ways))
        support, query = [], []
        for label, name in enumerate(chosen):
            picked = self.rng.sample(self.classes[name], self.shots + self.queries)
            support += [(p, label) for p in picked[:self.shots]]
            query += [(p, label) for p in picked[self.shots:]]
        return chosen, support, query


def subsample(classes: dict[str, list[str]], shots: int, seed: int) -> dict[str, list[str]]:
    """Keep `shots` random images per class (all images when shots <= 0)."""
    if shots <= 0:
        return classes
    rng = random.Random(seed)
    return {c: sorted(rng.sample(v, min(shots, len(v)))) for c, v in classes.items()}


def to_items(classes: dict[str, list[str]], names: Sequence[str]) -> list[tuple[str, int]]:
    return [(p, i) for i, name in enumerate(names) for p in classes[name]]
