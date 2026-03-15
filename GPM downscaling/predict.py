import os
import warnings

import joblib
import numpy as np
import pandas as pd
from autoxgb.predict import AutoXGBPredict

warnings.filterwarnings("ignore")


def predict(input_path, model_path="model_output", output_path=None):
    config = joblib.load(os.path.join(model_path, "axgb.config"))
    predictor = AutoXGBPredict(model_path=model_path)

    data = pd.read_csv(input_path)
    df = data[config.features].copy()
    df["id"] = range(len(df))

    final_preds = []
    for fold in range(config.num_folds):
        fold_df = df.copy(deep=True)
        if config.categorical_features:
            fold_df[config.categorical_features] = predictor.categorical_encoders[fold].transform(
                fold_df[config.categorical_features].values
            )
        preds = predictor.models[fold].predict(fold_df[config.features])
        final_preds.append(preds)

    data["downscaled_gpm"] = np.clip(np.mean(final_preds, axis=0), 0, None)

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_predicted.csv"
    data.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    predict(
        input_path="predict.csv",
        model_path="model_output",
        output_path="predict_result.csv",
    )
