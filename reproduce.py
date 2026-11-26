"""Train the three five-fold ResNet-18 ensembles and reproduce the equal-weight submission."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import (best_f1, build_model, check_files, folds_from_labels, get_loader, get_sampler, group_id,
                   predict_batches, preprocess, read_labels, read_test, training_plateau)


VARIANTS = {"raw": 34, "median": 34, "sharpen": 45}
SAMPLES = 3072
SUBMISSION = "submission_resnet18_raw_median_sharpen_aug.csv"


def split(records, groups):
    fit, valid = [], []
    for item in records:
        (valid if group_id(item[0]) in groups else fit).append(item)
    if any({label for _, label in part} != {0, 1} for part in (fit, valid)):
        raise ValueError("Each fold must have positives and negatives in both partitions")
    return fit, valid


def checkpoint_path(output, variant, fold):
    return output / variant / f"fold{fold}.pt"


def load_state(path, variant, groups):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("architecture") != "resnet18" or saved.get("validation_groups") != groups or
            bool(saved.get("median_filter")) != (variant == "median") or
            saved.get("sharpen_amount", 0.0) != (0.5 if variant == "sharpen" else 0.0)):
        raise ValueError(f"{path}: checkpoint does not match the requested configuration")
    return saved


def train_fold(records, args, variant, fold, groups):
    seed = 41 + fold
    random.seed(seed)
    torch.manual_seed(seed)
    fit, valid = split(records, groups)
    device = args.device
    loader_options = {"batch_size": args.batch_size, "workers": args.workers}
    learning_rate = 0.0007
    minimum_lr = learning_rate / 8
    validation_delta = 0.001
    sampler = get_sampler(fit, seed, SAMPLES)
    train_batches = get_loader(args.images, fit, variant, device, sampler, **loader_options)
    valid_batches = get_loader(args.images, valid, variant, device, **loader_options)
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, threshold=validation_delta,
        threshold_mode="abs", min_lr=minimum_lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses = []
    best_balanced_bce = float("inf")
    last_loss_improvement = floor_epochs = 0
    best, best_epoch = -1, 0
    checkpoint = checkpoint_path(args.output, variant, fold)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(".partial")
    columns = ("epoch", "train_bce", "val_bce", "val_positive_bce", "val_negative_bce",
               "val_balanced_bce", "val_f1", "threshold", "val_tp", "val_fp", "val_fn", "lr")
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with checkpoint.with_suffix(".csv").open("w", newline="") as history:
        writer = csv.writer(history)
        writer.writerow(columns)
        for epoch in range(1, VARIANTS[variant] + 1):
            model.train()
            loss_sum = torch.zeros((), device=device)
            rate = optimizer.param_groups[0]["lr"]
            for tiles, labels in train_batches:
                optimizer.zero_grad(set_to_none=True)
                inputs = preprocess(tiles, device, variant, augment=True)
                targets = labels.to(device, dtype=torch.float32, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext():
                    logits = model(inputs)
                loss = loss_fn(logits.float(), targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach() * len(labels)
            train_bce = loss_sum.item() / SAMPLES
            train_losses.append(train_bce)
            scores, labels, losses = predict_batches(model, valid_batches, device, variant, with_losses=True)
            positive_bce = float(losses[labels == 1].mean())
            negative_bce = float(losses[labels == 0].mean())
            balanced_bce = (positive_bce + negative_bce) / 2
            threshold, f1, tp, predicted = best_f1(scores, labels)
            writer.writerow((epoch, train_bce, float(losses.mean()), positive_bce, negative_bce, balanced_bce,
                             f1, threshold, tp, predicted - tp, int(labels.sum()) - tp, rate))
            history.flush()
            print(f"{variant} fold={fold} epoch={epoch:02} train_bce={train_bce:.4f} "
                  f"val_balanced_bce={balanced_bce:.4f} f1={f1:.4f}", flush=True)
            if f1 > best:
                best, best_epoch = f1, epoch
                torch.save({"architecture": "resnet18", "weights": model.state_dict(), "epoch": epoch,
                            "f1": f1, "threshold": threshold, "validation_groups": groups,
                            "median_filter": variant == "median",
                            "sharpen_amount": 0.5 if variant == "sharpen" else 0.0}, partial)
            scheduler.step(balanced_bce)
            if balanced_bce < best_balanced_bce - validation_delta:
                best_balanced_bce, last_loss_improvement = balanced_bce, epoch
            floor_epochs = floor_epochs + 1 if rate <= minimum_lr else 0
            if floor_epochs >= 5 and epoch - last_loss_improvement >= 5 and training_plateau(train_losses):
                break
    partial.replace(checkpoint)
    print(f"Saved {checkpoint} (epoch {best_epoch}, validation F1 {best:.4f})", flush=True)


def train_models(args, folds):
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--data", str(args.data),
            "--images", str(args.images), "--output", str(args.output), "--device", str(args.device),
            "--batch-size", str(args.batch_size), "--workers", str(args.workers), "train-fold"]
    variants = (args.variant,) if args.variant else VARIANTS
    for variant in variants:
        for fold, groups in enumerate(folds, start=1):
            if args.fold and fold != args.fold:
                continue
            path = checkpoint_path(args.output, variant, fold)
            if path.is_file():
                load_state(path, variant, groups)
                continue
            print(f"Training {variant} {fold}/5", flush=True)
            subprocess.run([*base, "--variant", variant, "--fold", str(fold)], check=True)


def scores_for_variant(records, test_paths, folds, args, variant):
    oof, labels, test_scores = [], [], []
    test_records = [(path, 0) for path in test_paths]
    device = args.device
    loader_options = {"batch_size": args.batch_size, "workers": args.workers}
    for fold, groups in enumerate(folds, start=1):
        _, valid = split(records, groups)
        state = load_state(checkpoint_path(args.output, variant, fold), variant, groups)
        model = build_model(device)
        model.load_state_dict(state["weights"])
        valid_batches = get_loader(args.images, valid, variant, device, **loader_options)
        scores, observed = predict_batches(model, valid_batches, device, variant)
        oof.append(scores.astype(np.float64))
        labels.extend(observed.tolist())
        test_batches = get_loader(args.images, test_records, variant, device, **loader_options)
        test_scores.append(predict_batches(model, test_batches, device, variant)[0])
        print(f"Scored {variant} {fold}/5", flush=True)
    mean = np.stack(test_scores).mean(axis=0)
    # The original blend read each float32 test mean back from a CSV column.
    return np.concatenate(oof), mean.astype(str).astype(np.float64), np.asarray(labels)


def write_submission(records, test_paths, folds, args):
    validation, test = [], []
    for variant in VARIANTS:
        valid_scores, test_scores, labels = scores_for_variant(records, test_paths, folds, args, variant)
        validation.append(valid_scores)
        test.append(test_scores)
    scores = (validation[0] + validation[1] + validation[2]) / 3
    mean = (test[0] + test[1] + test[2]) / 3
    threshold, _, _, _ = best_f1(scores, labels)
    flags = mean >= threshold

    fold_ids = np.concatenate([np.full(len(split(records, groups)[1]), fold)
                               for fold, groups in enumerate(folds, start=1)])
    predicted = np.empty(len(labels), dtype=bool)
    for fold in range(1, len(folds) + 1):
        held = fold_ids == fold
        cutoff, *_ = best_f1(scores[~held], labels[~held])
        predicted[held] = scores[held] >= cutoff
    tp = int(labels[predicted].sum())
    fp = int(predicted.sum()) - tp
    fn = int(labels.sum()) - tp

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    file = output / SUBMISSION
    temporary = file.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(
            (path.as_posix(), int(flag)) for path, flag in zip(test_paths, flags, strict=True)
        )
    temporary.replace(file)
    report = {"validation_f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
              "threshold": threshold, "test_images": len(test_paths), "test_positives": int(flags.sum()),
              "submission_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Submission written to {file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="directory of the contest manifests")
    parser.add_argument("--images", type=Path, help="directory containing W/ (default: DATA/SDNET2018)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="local output directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, help="data loader workers (default: 4 for CUDA, 0 otherwise)")
    modes = parser.add_subparsers(dest="mode")
    modes.add_parser("all", help="train all missing folds, then build the submission")
    modes.add_parser("predict", help="make the submission from locally trained checkpoints")
    for name in ("train", "train-fold"):
        command = modes.add_parser(name, help="train ResNet-18 checkpoints")
        command.add_argument("--variant", choices=VARIANTS, required=name == "train-fold")
        command.add_argument("--fold", type=int, choices=range(1, 6), required=name == "train-fold")
    args = parser.parse_args()
    args.mode = args.mode or "all"
    args.images = args.images or args.data / "SDNET2018"
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device auto or --device cpu.")
    args.device = torch.device(args.device)
    if args.workers is None:
        args.workers = 4 if args.device.type == "cuda" else 0
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    if args.device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.mode != "train-fold":
        print(f"Device: {args.device}", flush=True)
    records = read_labels(args.data / "train_label.csv")
    check_files(args.images, [path for path, _ in records])
    folds = folds_from_labels(records)
    if args.mode == "train-fold":
        train_fold(records, args, args.variant, args.fold, folds[args.fold - 1])
        return
    if args.mode == "train":
        train_models(args, folds)
        return
    test = read_test(args.data)
    check_files(args.images, test)
    if args.mode == "all":
        args.variant = args.fold = None
        train_models(args, folds)
    write_submission(records, test, folds, args)


if __name__ == "__main__":
    main()
