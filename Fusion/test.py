import os

import joblib
import numpy as np
import pandas as pd


def calculate_metrics(y_true, y_pred):
    O, P = np.array(y_true), np.array(y_pred)
    O_mean = np.mean(O)

    denom_cc = np.sqrt(np.sum((O - O_mean) ** 2) * np.sum((P - np.mean(P)) ** 2))
    cc  = float(np.sum((O - O_mean) * (P - np.mean(P))) / denom_cc) if denom_cc != 0 else 0.0
    nse = float(1 - np.sum((O - P) ** 2) / np.sum((O - O_mean) ** 2))
    me  = float(np.mean(P - O))
    bias = float(np.sum(P - O) / np.sum(O) * 100) if np.sum(O) != 0 else 0.0
    rmse = float(np.sqrt(np.mean((P - O) ** 2)))
    mae  = float(np.mean(np.abs(P - O)))

    return {"CC": cc, "NSE": nse, "ME": me, "Bias": bias, "RMSE": rmse, "MAE": mae}


def predict(input_path, model_path="model_output", output_path=None):
    config = joblib.load(os.path.join(model_path, "axgb.config"))
    categorical_encoders = joblib.load(os.path.join(model_path, "axgb.categorical_encoders"))

    data = pd.read_csv(input_path)
    df = data[config.features].copy()

    all_preds = []
    for fold in range(config.num_folds):
        fold_df = df.copy()
        if config.categorical_features:
            fold_df[config.categorical_features] = categorical_encoders[fold].transform(
                fold_df[config.categorical_features].values
            )
        all_preds.append(config.models[fold].predict(fold_df[config.features]))

    data["fused_precip"] = np.clip(np.mean(all_preds, axis=0), 0, None)

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_predicted.csv"
    data.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    if "truth" in data.columns:
        metrics = calculate_metrics(data["truth"].values, data["fused_precip"].values)
        print("\nEvaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    predict(
        input_path="predict.csv",
        model_path="model_output",
        output_path="predict_result.csv",
    )
