# compare_runs.py

#Find the latest DeepONet run directory under outputs/deeponet_run_*
#and launch the FNN runner on the same dataset, passing the found run_meta.json.
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def find_latest_deeponet_run(outputs_dir="outputs", pattern="deeponet_run_*"):
    out_path = Path(outputs_dir)
    if not out_path.exists():
        return None
    candidates = list(out_path.glob(pattern))
    if len(candidates) == 0:
        return None
    # Choose latest by modification time of run_meta.json if present, else by directory mtime
    best = None
    best_time = -1.0
    for d in candidates:
        run_meta = d / "run_meta.json"
        if run_meta.exists():
            t = run_meta.stat().st_mtime
        else:
            t = d.stat().st_mtime
        if t > best_time:
            best_time = t
            best = d
    return best

def parse_hidden(s):
    # parse comma-separated ints into tuple
    return tuple(int(x) for x in s.split(",")) if s else (256,256)

def main():
    parser = argparse.ArgumentParser(description="Find latest deeponet run and run FNN comparison")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Outputs directory containing deeponet_run_*")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda)")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs to train FNN")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for FNN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for FNN")
    parser.add_argument("--hidden", type=str, default="256,256", help="Comma-separated hidden layer sizes for FNN, e.g. '256,128'")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for FNN")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root folder for FNN outputs")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    # find latest deeponet run
    latest = find_latest_deeponet_run(outputs_dir=args.outputs_dir)
    if latest is None:
        print(f"No deeponet_run_* found under {args.outputs_dir}. Please run deeponet first.")
        sys.exit(2)
    run_meta_path = latest / "run_meta.json"
    if not run_meta_path.exists():
        print(f"Found candidate run directory {latest} but run_meta.json is missing. Please ensure deeponet saved run_meta.json.")
        sys.exit(3)

    print(f"[{datetime.now().isoformat()}] Found latest DeepONet run: {latest}")
    print(f"Using run_meta: {run_meta_path}")

    # read run_meta to get artifact path (dataset)
    with open(run_meta_path, "r") as fh:
        meta = json.load(fh)
    artifact_dataset = meta.get("artifact_dataset")
    if artifact_dataset is None or not os.path.exists(artifact_dataset):
        print("artifact_dataset not found in run_meta or file is missing:", artifact_dataset)
        print("You can still run fnn_runner by providing artifact path manually.")
        # continue but will error when calling fnn_runner
    else:
        print(f"Using dataset artifact: {artifact_dataset}")

    # import the runner function (must be run from project root or PYTHONPATH set)
    try:
        from fnn_runner import train_fnn_and_compare
    except Exception as e:
        print("Failed to import fnn_runner.train_fnn_and_compare. Make sure you run this script from project root")
        print("Error:", e)
        sys.exit(4)

    # prepare args for train_fnn_and_compare
    device = args.device
    if args.force_cpu:
        device = "cpu"
    # if user asked for cuda but torch not available, fallback
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU.")
                device = "cpu"
        except Exception:
            device = "cpu"

    hidden = parse_hidden(args.hidden)

    print("Launching FNN runner with parameters:")
    print("  device:", device)
    print("  epochs:", args.epochs)
    print("  batch_size:", args.batch_size)
    print("  lr:", args.lr)
    print("  hidden:", hidden)
    print("  dropout:", args.dropout)

    # call the runner
    run_dir, metrics = train_fnn_and_compare(
        artifact_dataset_path=artifact_dataset,
        deeponet_run_meta=str(run_meta_path),
        output_root=args.output_root,
        device=device,
        hidden=hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=12345,
    )

    print(f"[{datetime.now().isoformat()}] FNN runner finished. Outputs at: {run_dir}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()