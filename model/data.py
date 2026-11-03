from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .prompts import canonical_class_name


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class SceneSample:
    path: str
    class_name: str


def scan_image_folders(root: str | Path) -> list[SceneSample]:
    """Discover ``root/<class_name>/<image>`` or ``root/images/<class_name>``."""
    base = Path(root).expanduser().resolve()
    if (base / "images").is_dir():
        base = base / "images"
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    results: list[SceneSample] = []
    for folder in sorted(x for x in base.iterdir() if x.is_dir()):
        class_name = canonical_class_name(folder.name)
        for file in sorted(folder.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                results.append(SceneSample(str(file.resolve()), class_name))
    if not results:
        raise ValueError(f"No class-folder images were found in {base}")
    return results


def generate_manifest(samples: Sequence[SceneSample], *, seed: int = 42,
                      train_fraction: float = 0.6,
                      val_fraction: float = 0.2) -> dict:
        if not (0 < train_fraction < 1 and 0 <= val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    by_class: dict[str, list[SceneSample]] = defaultdict(list)
    for item in samples:
        by_class[item.class_name].append(item)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = {key: [] for key in ("train", "val", "test")}
    for cls in sorted(by_class):
        images = list(by_class[cls])
        if len(images) < 3:
            raise ValueError(f"Class {cls} needs at least three images for a 3-way split")
        rng.shuffle(images)
        n = len(images)
        n_train = min(max(1, int(n * train_fraction)), n - 2)
        n_val = min(max(1, int(n * val_fraction)), n - n_train - 1)
        for key, subset in (
            ("train", images[:n_train]),
            ("val", images[n_train:n_train + n_val]),
            ("test", images[n_train + n_val:]),
        ):
            groups[key].extend({"path": s.path, "class_name": s.class_name} for s in subset)
    return {"schema_version": 1, "seed": seed,
            "split_policy": "stratified_image_level",
            "train": groups["train"], "val": groups["val"], "test": groups["test"]}


def read_manifest(filename: str | Path) -> dict[str, list[SceneSample]]:
    path = Path(filename).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    seen: set[Path] = set()
    for split in ("train", "val", "test"):
        if split not in raw or not isinstance(raw[split], list):
            raise ValueError(f"Missing {split} split in {path}")
        items = []
        for record in raw[split]:
            if isinstance(record, dict):
                file, cls = record["path"], record["class_name"]
            elif isinstance(record, list) and len(record) == 3:
                file, _label, cls = record  # compatible with APPLeNet split triples
            else:
                raise ValueError(f"Invalid entry in {split}: {record!r}")
            resolved = Path(file).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            resolved = resolved.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Image listed in manifest is missing: {resolved}")
            if resolved in seen:
                raise ValueError(f"Image appears in multiple splits or twice: {resolved}")
            seen.add(resolved)
            items.append(SceneSample(str(resolved), canonical_class_name(cls)))
        splits[split] = items
    return splits


def write_manifest(manifest: dict, output: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(output).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def class_names(samples: Iterable[SceneSample]) -> list[str]:
    return sorted({sample.class_name for sample in samples})


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[SceneSample], classes: Sequence[str],
                 transform):
        self.samples = list(samples)
        self.classes = list(classes)
        self.lookup = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform
        unknown = set(s.class_name for s in self.samples) - set(self.lookup)
        if unknown:
            raise ValueError(f"Classes not present in classifier: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        with Image.open(item.path) as im:
            image = im.convert("RGB")
        return self.transform(image), self.lookup[item.class_name]


def image_transforms(*, train: bool, size: int = 224):
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision for data preprocessing") from exc
    if train:
        operations = [
            T.RandomResizedCrop(size, scale=(0.7, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    else:
        operations = [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
        ]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


class FewShotEpisodeSampler:
    """Draw independent K-shot train support + Q test query image tasks.

    A fresh *adaptation model* must be trained for every episode. Sharing a
    fitted model across tasks would leak labels into later test queries.
    """

    def __init__(self, train_samples: Sequence[SceneSample],
                 test_samples: Sequence[SceneSample], *, ways: int = 5,
                 shots: int = 1, queries: int = 15, seed: int = 42):
        self.train: dict[str, list[SceneSample]] = defaultdict(list)
        self.test: dict[str, list[SceneSample]] = defaultdict(list)
        for item in train_samples:
            self.train[item.class_name].append(item)
        for item in test_samples:
            self.test[item.class_name].append(item)
        self.ways, self.shots, self.queries = ways, shots, queries
        self.rng = random.Random(seed)
        self.eligible = sorted(
            key for key in self.train.keys() & self.test.keys()
            if len(self.train[key]) >= shots and len(self.test[key]) >= queries
        )
        if min(ways, shots, queries) < 1 or len(self.eligible) < ways:
            raise ValueError(
                f"Need {ways} classes with >= {shots} train and >= {queries} test images; "
                f"found {len(self.eligible)} eligible classes"
            )

    def sample(self) -> tuple[list[str], list[SceneSample], list[SceneSample]]:
        chosen = sorted(self.rng.sample(self.eligible, self.ways))
        support, query = [], []
        for cls in chosen:
            support.extend(self.rng.sample(self.train[cls], self.shots))
            query.extend(self.rng.sample(self.test[cls], self.queries))
        if {s.path for s in support} & {s.path for s in query}:
            raise RuntimeError("Support/query leakage detected")
        return chosen, support, query
