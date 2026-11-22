"""Command-line entry point for model training."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from house_price_predict_ml.data import DEFAULT_DATASET_PATH, load_dataset
from house_price_predict_ml.model import train_model

DEFAULT_MODEL_PATH = Path("models/house_price_model.joblib")


def main(argv: Sequence[str] | None = None) -> None:
    """Train and save the house-price model from the command line."""
    parser = argparse.ArgumentParser(description="House price prediction tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train a price prediction model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    train_parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    if args.command == "train":
        _, metrics = train_model(load_dataset(args.data), args.model_output)
        print(f"Mean absolute error: ${metrics['mae']:,.2f}")
        print(f"R² score: {metrics['r2']:.3f}")
        print(f"Saved model: {args.model_output}")
