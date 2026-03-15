import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from autoxgb import AutoXGB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def prepare_data(csv_path, test_size=0.25, random_state=42):
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nGPM statistics:\n{df['GPM'].describe()}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values detected:")
        print(missing[missing > 0])
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    df["id"] = range(len(df))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_path = "rainfall_train.csv"
    test_path = "rainfall_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    return train_path, test_path


def train_model(train_path, test_path=None, output_dir="model_output",
                num_folds=5, num_trials=100, time_limit=3600):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    axgb = AutoXGB(
        train_filename=train_path,
        output=output_dir,
        test_filename=test_path,
        task="regression",
        idx="id",
        targets=["GPM"],
        features=None,
        categorical_features=None,
        use_gpu=False,
        num_folds=num_folds,
        seed=42,
        num_trials=num_trials,
        time_limit=time_limit,
        fast=True,
    )

    print(f"\nTraining: {num_folds}-fold CV, {num_trials} Optuna trials, time limit {time_limit}s")
    axgb.train()
    print("Training complete.")
    return axgb


def evaluate_model(output_dir, test_path):
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    if not os.path.exists(pred_path):
        print("No test predictions found.")
        return

    y_true = pd.read_csv(test_path)["GPM"].values
    y_pred = pd.read_csv(pred_path)["GPM"].values

    results = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "MSE":  float(mean_squared_error(y_true, y_pred)),
    }

    print("\nTest set performance:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    out_path = os.path.join(output_dir, "test_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CTDF Stage 1: Daily GPM downscaling")
    parser.add_argument("--input",      type=str,   default="train_2024.csv")
    parser.add_argument("--output",     type=str,   default="model_output")
    parser.add_argument("--test-size",  type=float, default=0.25)
    parser.add_argument("--num-folds",  type=int,   default=5)
    parser.add_argument("--num-trials", type=int,   default=100)
    parser.add_argument("--time-limit", type=int,   default=3600)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    train_path, test_path = prepare_data(args.input, test_size=args.test_size)
    train_model(train_path, test_path, args.output,
                num_folds=args.num_folds,
                num_trials=args.num_trials,
                time_limit=args.time_limit)
    evaluate_model(args.output, test_path)


if __name__ == "__main__":
    main()
