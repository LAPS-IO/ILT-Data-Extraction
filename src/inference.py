import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from features import ILTDataset, get_model, get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EPS = 1e-8


def collect_images(root_dir):
    patterns = [os.path.join(root_dir, "**", f"*{ext}") for ext in IMAGE_EXTENSIONS]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern, recursive=True))
    file_list = sorted(set(file_list))
    if not file_list:
        raise FileNotFoundError(f"No images found under {root_dir}")
    return file_list


def load_labels(labels_path):
    with open(labels_path) as f:
        labels_dict = json.load(f)
    if not labels_dict:
        raise ValueError(f"Label file is empty: {labels_path}")
    return labels_dict


def load_model(model_path, num_classes, device):
    model = get_model(load=True, num_classes=num_classes)
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_log_adjustment(priors_csv, labels_dict, device):
    df = pd.read_csv(priors_csv)
    if df.shape[1] < 3:
        raise ValueError(
            "Priors CSV must have 3 columns: class, training_samples, weights"
        )

    class_col, train_col, weight_col = df.columns[:3]
    num_classes = max(labels_dict.values()) + 1
    train_counts = torch.zeros(num_classes, dtype=torch.float32)
    test_weights = torch.zeros(num_classes, dtype=torch.float32)

    for _, row in df.iterrows():
        class_name = str(row[class_col])
        if class_name not in labels_dict:
            continue
        idx = labels_dict[class_name]
        train_counts[idx] = float(row[train_col])
        test_weights[idx] = float(row[weight_col])

    if train_counts.sum() <= 0 or test_weights.sum() <= 0:
        raise ValueError("Training counts and weights must sum to a positive value")

    train_priors = (train_counts + EPS) / (train_counts + EPS).sum()
    test_priors = (test_weights + EPS) / (test_weights + EPS).sum()
    return (torch.log(test_priors) - torch.log(train_priors)).to(device)


def predict_adjusted(logits, log_adjustment):
    adjusted = logits + log_adjustment
    return F.softmax(adjusted, dim=-1)


def first_subfolder(images_root, file_path):
    """Return the first path component under images_root, or '_root' if none."""
    rel = Path(file_path).resolve().relative_to(Path(images_root).resolve())
    return rel.parts[0] if len(rel.parts) > 1 else "_root"


def save_predictions_by_subfolder(rows, output_dir):
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for subfolder, group in df.groupby("subfolder", sort=True):
        out_path = os.path.join(output_dir, f"{subfolder}.csv")
        group.drop(columns=["subfolder"]).to_csv(out_path, index=False)
        saved_paths.append(out_path)
        print(f"Saved {len(group)} predictions to {out_path}")
    return saved_paths


def run_inference(
    images_root,
    model_path,
    labels_path,
    priors_csv=None,
    output_path=None,
    batch_size=BATCH_SIZE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_dict = load_labels(labels_path)
    index_to_class = {idx: name for name, idx in labels_dict.items()}
    num_classes = max(labels_dict.values()) + 1

    model = load_model(model_path, num_classes, device)
    log_adjustment = None
    if priors_csv is not None:
        log_adjustment = load_log_adjustment(priors_csv, labels_dict, device)

    file_list = collect_images(images_root)
    transform = get_transforms()
    dataset = ILTDataset(file_list, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    rows = []
    with torch.no_grad():
        for data, paths in tqdm(loader, desc="Inference", unit="batch"):
            data = data.to(device)
            logits = model(data)

            if log_adjustment is not None:
                probs = predict_adjusted(logits, log_adjustment)
            else:
                probs = F.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            for i, path in enumerate(paths):
                row = {
                    "file_name": os.path.basename(path),
                    "path": path,
                    "subfolder": first_subfolder(images_root, path),
                }
                for rank in range(3):
                    class_idx = top_indices[i, rank].item()
                    row[f"top{rank + 1}_class"] = index_to_class[class_idx]
                    row[f"top{rank + 1}_probability"] = top_probs[i, rank].item()
                rows.append(row)

    output_dir = output_path or os.path.join(images_root, "predictions")
    return save_predictions_by_subfolder(rows, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt inference on all images under a root folder."
    )
    parser.add_argument(
        "images_root",
        help="Root folder containing images (searched recursively)",
    )
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument(
        "labels_json",
        help='Path to JSON label map, e.g. {"class_0": 0, "class_1": 1, ...}',
    )
    parser.add_argument(
        "priors_csv",
        nargs="?",
        default=None,
        help=(
            "Optional CSV with 3 columns: class, number of training samples, "
            "and per-class weights for prior correction"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output directory for CSV files, one per first-level subfolder "
            "(default: <images_root>/predictions)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inference (default: {BATCH_SIZE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        images_root=os.path.abspath(args.images_root),
        model_path=os.path.abspath(args.model_path),
        labels_path=os.path.abspath(args.labels_json),
        priors_csv=os.path.abspath(args.priors_csv) if args.priors_csv else None,
        output_path=os.path.abspath(args.output) if args.output else None,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
