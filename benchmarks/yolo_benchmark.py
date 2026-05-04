"""
YOLO Benchmark — image-level wildfire detection benchmark.

Runs a YOLO model on a wildfire dataset and produces thesis-ready charts:
- image-level accuracy
- precision / recall / F1
- confusion matrix
- YOLO speed breakdown
- inference time distribution
- detection confidence distribution
- detections per image

Ground truth is inferred from folder names:
    fire   -> positive class
    nofire -> negative class

Prediction rule:
    If YOLO detects at least one Fire or Smoke object, prediction = fire.
    Otherwise, prediction = no_fire.

Usage:
    uv run python benchmarks/yolo_benchmark.py \
        --images ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version \
        --model path/to/best.pt \
        --num-images 100

Example with charts:
    uv run python benchmarks/yolo_benchmark.py \
        --images ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version \
        --model path/to/best.pt \
        --num-images 100 \
        --output results/yolo \
        --charts charts/yolo
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

POSITIVE_YOLO_CLASSES = {"fire", "smoke"}


# ── Dataset loading ──────────────────────────────────────────────────

def infer_ground_truth(image_path: Path):
    """
    Infer ground truth from folder names.

    Expected dataset structure can be like:
        train/fire/image.jpg
        train/nofire/image.jpg
        test/fire/image.jpg
        val/nofire/image.jpg
    """
    parts = {p.lower() for p in image_path.parts}

    if "nofire" in parts or "no_fire" in parts or "nonfire" in parts or "non_fire" in parts:
        return "no_fire"

    if "fire" in parts:
        return "fire"

    return None


def infer_split(image_path: Path):
    parts = [p.lower() for p in image_path.parts]

    for split in ["train", "val", "valid", "validation", "test"]:
        if split in parts:
            if split in {"valid", "validation"}:
                return "val"
            return split

    return "unknown"


def collect_images(images_dir: str):
    """Collect all images with usable fire/no_fire ground truth."""
    root = Path(images_dir)
    images = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        gt = infer_ground_truth(path)
        if gt is None:
            continue

        images.append(path)

    return images


def sample_images(images, num_images=None, seed=42, balanced=False):
    """Sample images reproducibly. Optional balanced fire/no_fire sampling."""
    if num_images is None or num_images >= len(images):
        return images

    rng = random.Random(seed)

    if not balanced:
        return rng.sample(images, num_images)

    fire_images = [p for p in images if infer_ground_truth(p) == "fire"]
    nofire_images = [p for p in images if infer_ground_truth(p) == "no_fire"]

    half = num_images // 2
    remainder = num_images - (half * 2)

    sampled = []

    sampled.extend(rng.sample(fire_images, min(half + remainder, len(fire_images))))
    sampled.extend(rng.sample(nofire_images, min(half, len(nofire_images))))

    rng.shuffle(sampled)
    return sampled


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(rows):
    tp = fp = tn = fn = 0

    for r in rows:
        gt = r["ground_truth"]
        pred = r["yolo_prediction"]

        if gt == "fire" and pred == "fire":
            tp += 1
        elif gt == "no_fire" and pred == "fire":
            fp += 1
        elif gt == "no_fire" and pred == "no_fire":
            tn += 1
        elif gt == "fire" and pred == "no_fire":
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def group_by_model(rows):
    groups = {}

    for row in rows:
        model = row["model_name"]
        groups.setdefault(model, [])
        groups[model].append(row)

    return groups


def safe_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ── YOLO benchmarking ────────────────────────────────────────────────

def run_yolo_on_image(model, image_path, model_name, args):
    """Run YOLO on a single image and return one benchmark row."""
    start = time.perf_counter()

    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    wall_time_ms = (time.perf_counter() - start) * 1000

    names = result.names
    boxes = result.boxes

    total_boxes = 0
    positive_boxes = 0
    fire_boxes = 0
    smoke_boxes = 0
    confidences = []
    positive_confidences = []

    if boxes is not None and len(boxes) > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()

        for class_id, conf in zip(class_ids, confs):
            class_name = str(names.get(class_id, class_id)).lower()

            total_boxes += 1
            confidences.append(conf)

            if class_name == "fire":
                fire_boxes += 1

            if class_name == "smoke":
                smoke_boxes += 1

            if class_name in POSITIVE_YOLO_CLASSES:
                positive_boxes += 1
                positive_confidences.append(conf)

    yolo_prediction = "fire" if positive_boxes > 0 else "no_fire"
    ground_truth = infer_ground_truth(image_path)
    correct = yolo_prediction == ground_truth

    preprocess_ms = result.speed.get("preprocess", 0.0)
    inference_ms = result.speed.get("inference", 0.0)
    postprocess_ms = result.speed.get("postprocess", 0.0)
    yolo_total_ms = preprocess_ms + inference_ms + postprocess_ms

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "split": infer_split(image_path),
        "ground_truth": ground_truth,
        "yolo_prediction": yolo_prediction,
        "correct": str(correct),
        "total_boxes": total_boxes,
        "positive_boxes": positive_boxes,
        "fire_boxes": fire_boxes,
        "smoke_boxes": smoke_boxes,
        "max_confidence": max(positive_confidences) if positive_confidences else "",
        "mean_confidence": float(np.mean(positive_confidences)) if positive_confidences else "",
        "max_any_confidence": max(confidences) if confidences else "",
        "mean_any_confidence": float(np.mean(confidences)) if confidences else "",
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "yolo_total_ms": yolo_total_ms,
        "wall_time_ms": wall_time_ms,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device if args.device is not None else "auto",
    }


def write_csv(rows, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"yolo_benchmark_{timestamp}.csv"

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def load_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Chart helpers ────────────────────────────────────────────────────

def save_chart(fig, charts_dir, filename):
    path = Path(charts_dir)
    path.mkdir(parents=True, exist_ok=True)

    full_path = path / filename
    fig.savefig(full_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved {full_path}")


def add_bar_labels(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ── Charts ───────────────────────────────────────────────────────────

def chart_accuracy_metrics(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for model in models:
        m = compute_metrics(groups[model])
        accuracies.append(m["accuracy"] * 100)
        precisions.append(m["precision"] * 100)
        recalls.append(m["recall"] * 100)
        f1s.append(m["f1"] * 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.2

    ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
    ax.bar(x - 0.5 * width, precisions, width, label="Precision")
    ax.bar(x + 0.5 * width, recalls, width, label="Recall")
    ax.bar(x + 1.5 * width, f1s, width, label="F1")

    ax.set_ylabel("Score (%)")
    ax.set_title("YOLO Image-Level Detection Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_chart(fig, charts_dir, "yolo_accuracy_metrics.png")


def chart_confusion_matrix(rows, charts_dir):
    groups = group_by_model(rows)

    for model, model_rows in groups.items():
        m = compute_metrics(model_rows)

        matrix = np.array([
            [m["tp"], m["fn"]],
            [m["fp"], m["tn"]],
        ])

        fig, ax = plt.subplots(figsize=(5.5, 5))

        im = ax.imshow(matrix)

        ax.set_title(f"YOLO Confusion Matrix — {model}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Ground truth")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fire", "no_fire"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["fire", "no_fire"])

        labels = [
            ["TP", "FN"],
            ["FP", "TN"],
        ]

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{matrix[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        safe_model = model.replace("/", "_").replace(":", "_")
        filename = (
            "yolo_confusion_matrix.png"
            if len(groups) == 1
            else f"yolo_confusion_matrix_{safe_model}.png"
        )

        save_chart(fig, charts_dir, filename)


def chart_confusion_counts(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    tps = []
    fps = []
    tns = []
    fns = []

    for model in models:
        m = compute_metrics(groups[model])
        tps.append(m["tp"])
        fps.append(m["fp"])
        tns.append(m["tn"])
        fns.append(m["fn"])

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.2

    ax.bar(x - 1.5 * width, tps, width, label="TP: detected fire")
    ax.bar(x - 0.5 * width, tns, width, label="TN: rejected no-fire")
    ax.bar(x + 0.5 * width, fps, width, label="FP: false alarm")
    ax.bar(x + 1.5 * width, fns, width, label="FN: missed fire")

    ax.set_ylabel("Image count")
    ax.set_title("YOLO Confusion Counts by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    save_chart(fig, charts_dir, "yolo_confusion_counts.png")


def chart_speed_breakdown(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    preprocess = []
    inference = []
    postprocess = []

    for model in models:
        model_rows = groups[model]

        preprocess.append(np.median([safe_float(r["preprocess_ms"]) for r in model_rows]))
        inference.append(np.median([safe_float(r["inference_ms"]) for r in model_rows]))
        postprocess.append(np.median([safe_float(r["postprocess_ms"]) for r in model_rows]))

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.55

    ax.bar(x, preprocess, width, label="Preprocess")
    ax.bar(x, inference, width, bottom=preprocess, label="Inference")
    ax.bar(
        x,
        postprocess,
        width,
        bottom=[p + i for p, i in zip(preprocess, inference)],
        label="Postprocess",
    )

    totals = [p + i + po for p, i, po in zip(preprocess, inference, postprocess)]

    for i, total in enumerate(totals):
        ax.text(i, total, f"{total:.1f} ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Time (ms)")
    ax.set_title("YOLO Speed Breakdown per Image — Median")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_chart(fig, charts_dir, "yolo_speed_breakdown.png")


def chart_total_time(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    yolo_medians = []
    wall_medians = []
    yolo_stds = []

    for model in models:
        model_rows = groups[model]

        yolo_times = [safe_float(r["yolo_total_ms"]) for r in model_rows]
        wall_times = [safe_float(r["wall_time_ms"]) for r in model_rows]

        yolo_medians.append(np.median(yolo_times))
        wall_medians.append(np.median(wall_times))
        yolo_stds.append(np.std(yolo_times))

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        yolo_medians,
        width,
        yerr=yolo_stds,
        capsize=4,
        label="YOLO-reported total",
    )

    bars2 = ax.bar(
        x + width / 2,
        wall_medians,
        width,
        label="Measured wall time",
    )

    ax.set_ylabel("Time per image (ms)")
    ax.set_title("YOLO Total Inference Time per Image")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    add_bar_labels(ax, bars1, "{:.0f}")
    add_bar_labels(ax, bars2, "{:.0f}")

    save_chart(fig, charts_dir, "yolo_total_time.png")


def chart_time_distribution(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    data = []

    for model in models:
        data.append([safe_float(r["yolo_total_ms"]) for r in groups[model]])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(data, labels=models, patch_artist=True)

    ax.set_ylabel("YOLO total time (ms)")
    ax.set_title("YOLO Inference Time Distribution")
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=25, ha="right")

    save_chart(fig, charts_dir, "yolo_time_distribution.png")


def chart_detection_counts_by_truth(rows, charts_dir):
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    fire_counts = []
    nofire_counts = []

    for model in models:
        model_rows = groups[model]

        fire_rows = [r for r in model_rows if r["ground_truth"] == "fire"]
        nofire_rows = [r for r in model_rows if r["ground_truth"] == "no_fire"]

        fire_counts.append(
            np.mean([safe_int(r["positive_boxes"]) for r in fire_rows])
            if fire_rows else 0
        )

        nofire_counts.append(
            np.mean([safe_int(r["positive_boxes"]) for r in nofire_rows])
            if nofire_rows else 0
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, fire_counts, width, label="Ground truth: fire")
    ax.bar(x + width / 2, nofire_counts, width, label="Ground truth: no_fire")

    ax.set_ylabel("Average positive detections per image")
    ax.set_title("YOLO Fire/Smoke Detections by Ground Truth")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_chart(fig, charts_dir, "yolo_detections_by_truth.png")


def chart_confidence_distribution(rows, charts_dir):
    groups = group_by_model(rows)

    for model, model_rows in groups.items():
        fire_conf = [
            safe_float(r["max_confidence"])
            for r in model_rows
            if r["ground_truth"] == "fire" and r["max_confidence"] != ""
        ]

        nofire_conf = [
            safe_float(r["max_confidence"])
            for r in model_rows
            if r["ground_truth"] == "no_fire" and r["max_confidence"] != ""
        ]

        if not fire_conf and not nofire_conf:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.boxplot(
            [fire_conf, nofire_conf],
            labels=["fire images", "no_fire images"],
            patch_artist=True,
        )

        ax.set_ylabel("Maximum fire/smoke confidence")
        ax.set_title(f"YOLO Detection Confidence — {model}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

        safe_model = model.replace("/", "_").replace(":", "_")
        filename = (
            "yolo_confidence_distribution.png"
            if len(groups) == 1
            else f"yolo_confidence_distribution_{safe_model}.png"
        )

        save_chart(fig, charts_dir, filename)


def chart_speed_vs_boxes(rows, charts_dir):
    groups = group_by_model(rows)

    for model, model_rows in groups.items():
        x = [safe_int(r["total_boxes"]) for r in model_rows]
        y = [safe_float(r["yolo_total_ms"]) for r in model_rows]

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(x, y, alpha=0.7)

        ax.set_xlabel("Total YOLO boxes")
        ax.set_ylabel("YOLO total time (ms)")
        ax.set_title(f"YOLO Speed vs Number of Detections — {model}")
        ax.grid(alpha=0.25)

        safe_model = model.replace("/", "_").replace(":", "_")
        filename = (
            "yolo_speed_vs_boxes.png"
            if len(groups) == 1
            else f"yolo_speed_vs_boxes_{safe_model}.png"
        )

        save_chart(fig, charts_dir, filename)


# ── Summary ──────────────────────────────────────────────────────────

def print_summary(rows):
    groups = group_by_model(rows)

    print()
    print(
        f"{'Model':<22} "
        f"{'N':>5} "
        f"{'Acc':>7} "
        f"{'Prec':>7} "
        f"{'Rec':>7} "
        f"{'F1':>7} "
        f"{'MCC':>7} "
        f"{'Med ms':>9} "
        f"{'FPS':>8}"
    )
    print("─" * 89)

    for model in sorted(groups.keys()):
        model_rows = groups[model]
        m = compute_metrics(model_rows)

        median_ms = np.median([safe_float(r["yolo_total_ms"]) for r in model_rows])
        fps = 1000 / median_ms if median_ms > 0 else 0

        print(
            f"{model:<22} "
            f"{m['total']:>5} "
            f"{m['accuracy'] * 100:>6.1f}% "
            f"{m['precision'] * 100:>6.1f}% "
            f"{m['recall'] * 100:>6.1f}% "
            f"{m['f1'] * 100:>6.1f}% "
            f"{m['mcc']:>7.2f} "
            f"{median_ms:>8.1f} "
            f"{fps:>8.1f}"
        )

        print(
            f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}"
        )

    print()


def save_summary(rows, charts_dir):
    path = Path(charts_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_path = path / "yolo_summary.txt"

    groups = group_by_model(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        for model in sorted(groups.keys()):
            model_rows = groups[model]
            m = compute_metrics(model_rows)

            median_pre = np.median([safe_float(r["preprocess_ms"]) for r in model_rows])
            median_inf = np.median([safe_float(r["inference_ms"]) for r in model_rows])
            median_post = np.median([safe_float(r["postprocess_ms"]) for r in model_rows])
            median_total = np.median([safe_float(r["yolo_total_ms"]) for r in model_rows])
            fps = 1000 / median_total if median_total > 0 else 0

            f.write(f"Model: {model}\n")
            f.write(f"Images: {m['total']}\n")
            f.write(f"Accuracy: {m['accuracy'] * 100:.2f}%\n")
            f.write(f"Precision: {m['precision'] * 100:.2f}%\n")
            f.write(f"Recall: {m['recall'] * 100:.2f}%\n")
            f.write(f"F1: {m['f1'] * 100:.2f}%\n")
            f.write(f"MCC: {m['mcc']:.3f}\n")
            f.write(f"TP: {m['tp']}\n")
            f.write(f"FP: {m['fp']}\n")
            f.write(f"TN: {m['tn']}\n")
            f.write(f"FN: {m['fn']}\n")
            f.write(f"Median preprocess time: {median_pre:.2f} ms\n")
            f.write(f"Median inference time: {median_inf:.2f} ms\n")
            f.write(f"Median postprocess time: {median_post:.2f} ms\n")
            f.write(f"Median total YOLO time: {median_total:.2f} ms\n")
            f.write(f"Approx. FPS: {fps:.2f}\n")
            f.write("\n")

    print(f"  Saved {summary_path}")


def generate_charts(rows, charts_dir):
    print("\nGenerating YOLO charts...")

    chart_accuracy_metrics(rows, charts_dir)
    chart_confusion_matrix(rows, charts_dir)
    chart_confusion_counts(rows, charts_dir)
    chart_speed_breakdown(rows, charts_dir)
    chart_total_time(rows, charts_dir)
    chart_time_distribution(rows, charts_dir)
    chart_detection_counts_by_truth(rows, charts_dir)
    chart_confidence_distribution(rows, charts_dir)
    chart_speed_vs_boxes(rows, charts_dir)
    save_summary(rows, charts_dir)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO wildfire detection")

    parser.add_argument(
        "--images",
        required=True,
        help="Path to wildfire dataset root",
    )

    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="YOLO model path(s), e.g. path/to/best.pt",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to sample",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling",
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample approximately equal fire and no_fire images",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO IoU threshold",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. cpu, 0, mps. Leave empty for auto.",
    )

    parser.add_argument(
        "--output",
        default="results/yolo",
        help="Directory for CSV output",
    )

    parser.add_argument(
        "--charts",
        default="charts/yolo",
        help="Directory for chart output",
    )

    parser.add_argument(
        "--csv-only",
        help="Skip running YOLO and generate charts from an existing CSV",
    )

    args = parser.parse_args()

    if args.csv_only:
        rows = load_csv(args.csv_only)
        print(f"Loaded {len(rows)} rows from {args.csv_only}")
        generate_charts(rows, args.charts)
        print_summary(rows)
        return

    images = collect_images(args.images)

    if not images:
        raise SystemExit(f"No labelled images found in {args.images}")

    sampled_images = sample_images(
        images,
        num_images=args.num_images,
        seed=args.seed,
        balanced=args.balanced,
    )

    fire_count = sum(1 for p in sampled_images if infer_ground_truth(p) == "fire")
    nofire_count = sum(1 for p in sampled_images if infer_ground_truth(p) == "no_fire")

    print(
        f"Loaded {len(images)} labelled images. "
        f"Benchmarking {len(sampled_images)} images "
        f"({fire_count} fire, {nofire_count} no_fire)."
    )

    all_rows = []

    for model_arg in args.model:
        model_path = Path(model_arg)
        model_name = model_path.stem if model_path.suffix == ".pt" else model_arg

        print(f"\n=== YOLO benchmark: {model_name} ===")

        model = YOLO(model_arg)

        # Warmup
        print("Warming up YOLO model...")
        _ = model.predict(
            source=str(sampled_images[0]),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        for idx, image_path in enumerate(sampled_images, start=1):
            row = run_yolo_on_image(model, image_path, model_name, args)
            all_rows.append(row)

            symbol = "✓" if row["correct"] == "True" else "✗"

            print(
                f"[{idx}/{len(sampled_images)}] "
                f"{image_path.name} -> {row['yolo_prediction']} ({symbol}) | "
                f"boxes={row['positive_boxes']} | "
                f"{safe_float(row['yolo_total_ms']):.1f} ms"
            )

    csv_path = write_csv(all_rows, args.output)
    print(f"\nSaved benchmark CSV to {csv_path}")

    generate_charts(all_rows, args.charts)
    print_summary(all_rows)

    print(f"Done. Charts saved to {args.charts}/")


if __name__ == "__main__":
    main()