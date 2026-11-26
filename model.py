"""ResNet-18 and image preprocessing for the P-con classification submission."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18
from torchvision.transforms.functional import gaussian_blur


def wall_path(value):
    path = PurePosixPath(value)
    if len(path.parts) != 3 or path.parts[:2] not in (("W", "CW"), ("W", "UW")) or path.suffix.lower() != ".jpg":
        raise ValueError(f"Expected a wall JPG under W/CW or W/UW, got {value!r}")
    return path


def require_unique(paths, source):
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"{source}: expected a nonempty list of unique image paths")


def read_labels(filename):
    records = []
    with filename.open(encoding="utf-8-sig", newline="") as file:
        for line, row in enumerate(csv.reader(file), 1):
            if len(row) != 2 or row[1] not in ("0", "1"):
                raise ValueError(f"{filename}:{line}: expected path and binary label")
            records.append((wall_path(row[0]), int(row[1])))
    require_unique([path for path, _ in records], filename)
    return records


def read_test(manifests):
    filename = manifests / "test.txt"
    paths = [wall_path(line.strip()) for line in filename.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require_unique(paths, filename)
    template = [path for path, _ in read_labels(manifests / "sample_submission.csv")]
    if paths != template:
        raise ValueError("test.txt and sample_submission.csv differ in image paths or order")
    return paths


def check_files(root, paths):
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} images missing under {root}, including {missing[0]}")


def group_id(path):
    return path.stem.split("-", 1)[0]


def folds_from_labels(records):
    """Rebuild the original five-fold partition using only the provided labels."""
    by_group = defaultdict(Counter)
    for path, label in records:
        by_group[group_id(path)][label, path.parent.name] += 1
    groups = sorted(by_group)
    if len(groups) < 5 or len(groups) % 5:
        raise ValueError("Expected the competition's five equally sized sets of source photos")
    holdout_size = len(groups) // 5
    totals = sum(by_group.values(), Counter())
    total_positive = sum(count for (label, _), count in totals.items() if label)
    total_negative = len(records) - total_positive

    first_rng = random.Random(42)
    first_cost, first = float("inf"), None
    for _ in range(512):
        selected = first_rng.sample(groups, holdout_size)
        counts = sum((by_group[group] for group in selected), Counter())
        positives = sum(count for (label, _), count in counts.items() if label)
        negatives = sum(counts.values()) - positives
        if not (0 < positives < total_positive and 0 < negatives < total_negative):
            continue
        cost = sum((counts[key] / total - holdout_size / len(groups)) ** 2 for key, total in totals.items())
        if cost < first_cost:
            first_cost, first = cost, set(selected)
    if first is None:
        raise ValueError("Unable to create a first validation fold with both classes")

    remaining = sorted(set(groups) - first)
    stratum_counts = np.array([[by_group[g][label, wall] for wall in ("UW", "CW") for label in (0, 1)]
                              for g in remaining])
    target = stratum_counts.sum(axis=0) / 4
    if np.any(target == 0):
        raise ValueError("Expected positive and negative labels for both kinds of walls")
    rng = random.Random(42)
    best_cost, partitions = float("inf"), None
    for _ in range(10000):
        candidate = np.asarray(rng.sample(range(len(remaining)), len(remaining))).reshape(4, -1)
        ratios = stratum_counts[candidate].sum(axis=1) / target
        cost = float(np.square(ratios - 1).sum())
        if cost < best_cost:
            best_cost, partitions = cost, candidate
    return [sorted(first)] + [sorted(remaining[i] for i in fold) for fold in partitions]


class WallTiles(Dataset):
    def __init__(self, root, records, *, median, augment):
        self.root = root
        self.records = records
        self.median = median
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(self.root / path) as stored:
            if stored.size != (256, 256):
                raise ValueError(f"{path}: expected 256x256, got {stored.size}")
            image = stored.convert("RGB")
        if self.augment:
            rotation = random.choice((None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                      Image.Transpose.ROTATE_270))
            if rotation is not None:
                image = image.transpose(rotation)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.35, 0.85)))
        pixels = np.array(image)
        if self.median:
            from cv2 import medianBlur

            pixels = medianBlur(pixels, 3)
        return torch.from_numpy(pixels).permute(2, 0, 1), label


def get_loader(images, records, variant, device, sampler=None, *, batch_size=32, workers=4):
    return DataLoader(WallTiles(images, records, median=variant == "median", augment=sampler is not None),
                      batch_size=batch_size, sampler=sampler, num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=device.type == "cuda")


def get_sampler(records, seed, samples):
    probabilities = {"positive": 0.25, "cracked": 0.34, "other": 0.41}
    kinds = ["positive" if label else "cracked" if path.parent.name == "CW" else "other" for path, label in records]
    counts = Counter(kinds)
    return WeightedRandomSampler([probabilities[kind] / counts[kind] for kind in kinds], samples, replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def build_model(device):
    model = resnet18(weights=None, num_classes=1)
    model.fc = nn.Sequential(model.fc, nn.Flatten(start_dim=0))
    memory_format = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    return model.to(device, memory_format=memory_format)


def preprocess(tiles, device, variant, *, augment=False):
    x = tiles.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda").div_(255)
    if augment:
        shape = (len(x), 1, 1, 1)
        gain = torch.rand(shape, device=device) * (2 * 0.20) + (1 - 0.20)
        offset = torch.rand(shape, device=device) * (2 * 0.12) - 0.12
        x = ((x - 0.5) * gain + 0.5 + offset).clamp_(0, 1)
    memory_format = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    x = x.mul_(2).sub_(1).contiguous(memory_format=memory_format)
    if variant == "sharpen":
        amount = 0.5
        if augment:
            variation = torch.rand((len(x), 1, 1, 1), device=device) * 2 - 1
            amount = amount + 0.3 * variation
        x = (x + amount * (x - gaussian_blur(x, [9, 9], [1.5, 1.5]))).clamp_(-1, 1)
        x = x.contiguous(memory_format=memory_format)
    return x


@torch.inference_mode()
def predict_batches(model, batches, device, variant, *, with_losses=False):
    model.eval()
    scores, labels, losses = [], [], []
    for tiles, targets in batches:
        logits = model(preprocess(tiles, device, variant))
        scores.append(logits.sigmoid().cpu().numpy())
        labels.append(targets.numpy())
        if with_losses:
            targets = targets.to(device=device, dtype=logits.dtype)
            losses.append(F.binary_cross_entropy_with_logits(logits, targets, reduction="none").cpu().numpy())
    if with_losses:
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(losses)
    return np.concatenate(scores), np.concatenate(labels)


def best_f1(scores, labels):
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    boundaries = np.flatnonzero(np.r_[ranked[:-1] != ranked[1:], True])
    tp = labels[order].cumsum()[boundaries]
    count = boundaries + 1
    f1 = 2 * tp / (count + labels.sum())
    best = int(f1.argmax())
    return float(ranked[boundaries[best]]), float(f1[best]), int(tp[best]), int(count[best])


def training_plateau(losses):
    if len(losses) < 8:
        return False
    previous = np.median(losses[-8:-4])
    recent = np.median(losses[-4:])
    return abs(recent - previous) <= max(0.002, 0.05 * previous)
