import os
import argparse
import logging

import gcsfs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(csv_path: str):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Expecting adult-income-fixed.csv style columns
    # ['age','workclass','fnlwgt','education','education_num',
    #  'marital_status','occupation','relationship','race','sex',
    #  'capital_gain','capital_loss','hours_per_week','native_country','income']
    if "income" not in df.columns:
        raise ValueError("Expected 'income' column in CSV as target label.")

    X = df.drop(columns=["income"])
    y = df["income"]

    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    return model


def main(args):
    # For Vertex AI custom training, training data will be in a GCS path
    # but the container sees it as a local path if we use the gsutil download
    # step or mount GCS. For simplicity, we'll read directly via pandas+gcsfs

    logger.info("Starting training job")

    # args.train_data must be a GCS URI: gs://adult-income/adult-income-fixed.csv
    if not args.train_data.startswith("gs://"):
        raise ValueError("train_data must be a GCS URI like gs://bucket/file.csv")

    fs = gcsfs.GCSFileSystem()
    with fs.open(args.train_data) as f:
        df = pd.read_csv(f)

    if "income" not in df.columns:
        raise ValueError("Expected 'income' column in CSV as target label.")

    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)

    logger.info("Fitting model...")
    model.fit(X_train, y_train)

    logger.info("Evaluating model...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Validation accuracy: {acc:.4f}")

    model_dir = os.environ.get("AIP_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AIP_MODEL_DIR is not set; cannot save model to GCS.")

    fs = gcsfs.GCSFileSystem()

    model_path = f"{model_dir.rstrip('/')}/model.joblib"
    metrics_path = f"{model_dir.rstrip('/')}/metrics.txt"

    logger.info(f"Saving model to {model_path}")
    with fs.open(model_path, "wb") as f:
        joblib.dump(model, f)

    logger.info(f"Saving metrics to {metrics_path}")
    with fs.open(metrics_path, "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
    logger.info("Training job completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="GCS path to training CSV, e.g. gs://adult-income/adult-income-fixed.csv",
    )
    args = parser.parse_args()
    main(args)