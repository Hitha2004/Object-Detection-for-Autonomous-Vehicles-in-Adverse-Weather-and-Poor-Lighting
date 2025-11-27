"""
train_yolov8.py
A simple, clear training script for YOLOv8 using Ultralytics Python API.

Usage examples:
    # Basic
    python train_yolov8.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16

    # Use CUDA device 0
    python train_yolov8.py --data data.yaml --device 0 --epochs 100

Requirements: ultralytics (pip install ultralytics), torch, and GPU drivers for CUDA (if using GPU).
"""

import argparse
from ultralytics import YOLO
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset")
    p.add_argument("--data", type=str, default="data.yaml", help="Path to dataset YAML")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Backbone pretrained weights or model (yolov8n.pt/yolov8s.pt/...)")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--device", type=str, default="cuda" , help="Device: 'cpu' or 'cuda' or 'cuda:0' etc.")
    p.add_argument("--save_dir", type=str, default="runs/train", help="Directory to save results")
    p.add_argument("--project", type=str, default="YOLOv8_custom", help="Ultralytics project name (for organized runs)")
    p.add_argument("--name", type=str, default=None, help="Experiment name (if None uses timestamp)")
    p.add_argument("--resume", action="store_true", help="Resume from last training checkpoint")
    p.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (0 to disable)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_best", action="store_true", help="Save only best.pt (automatically handled by Ultralytics)")
    p.add_argument("--adam", action="store_true", help="Use Adam optimizer (default is SGD/AdamW whichever Ultralytics chooses)")
    return p.parse_args()

def main(args):
    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model (pretrained)
    model = YOLO(args.model)  # load a pretrained model (e.g. yolov8n.pt)

    # Training keyword args map — these match the ultralytics model.train() parameters.
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgs": args.imgsz,        # shorthand param name in newer versions
        "batch": args.batch,
        "lr0": args.lr,           # initial learning rate
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "workers": args.workers,
        "resume": args.resume,
        "seed": args.seed,
        # "patience": args.patience,  # Ultralytics has built-in early stopping options in yaml/hyp; if supported, pass here
        # "optimizer": "Adam" if args.adam else "SGD",
        # optionally add more Ultralytics-specific args as needed
    }

    # Optional hyperparameters dictionary — you can override many internal hyperparams.
    # Example: train_kwargs["hyp"] = "hyp_custom.yaml"  # or dict with hyperparams
    # If you want to specify custom hyperparameters, create a YAML hyp file and pass its path here.

    print("Starting training with config:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    # Train
    results = model.train(**train_kwargs)

    # results is an ultralytics.engine.results.Results object
    print("Training complete. Results object keys/attrs:")
    try:
        print(f"  best_fitness: {results.best_fitness}")
        print(f"  best_results: {results.best_results}")
        print(f"  path: {results.path}")
    except Exception:
        # Different ultralytics versions expose different fields
        print("  (results object fields may differ by ultralytics version)")

    print("Done. Trained weights are in the results folder (project/name).")

if __name__ == "__main__":
    args = parse_args()
    main(args)
