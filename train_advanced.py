"""
Train advanced ML models for the Problem 8 mini-project.

Models:
- LightGBM
- Random Forest
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (FFNN)

The script evaluates both feature matrices and both target variables.
A held-out test split is used for the final AUROC; CV on the training split
is used for model selection/tuning. This avoids using the test set for tuning.

Usage:
    python train_advanced.py
    python train_advanced.py --dataset knn --outcome drop
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_CV = 5


def load_data(dataset_name, outcome_name):
    if dataset_name == "knn":
        path = "data/X_knn_933.csv"
    elif dataset_name == "binned":
        path = "data/X_binned_1367.csv"
    else:
        raise ValueError("dataset must be 'knn' or 'binned'")

    X = pd.read_csv(path)
    y_df = pd.read_csv("data/y_targets.csv")

    target_col = {"drop": "y_drop", "threshold": "y_threshold"}[outcome_name]
    y = y_df[target_col].astype(int).to_numpy()

    # The current featurization should already be numeric, but this keeps
    # the training script robust to an accidental non-numeric column.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    return X, y


def build_models():
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            objective="binary",
            verbosity=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "FFNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def cv_auc(model, X_train, y_train, n_splits=DEFAULT_CV):
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scores = []

    for tr_idx, va_idx in skf.split(X_train, y_train):
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        prob = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y_train[va_idx], prob))

    return float(np.mean(scores)), float(np.std(scores))


def run_one(dataset_name, outcome_name, cv_folds):
    X, y = load_data(dataset_name, outcome_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = []
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for model_name, model in build_models().items():
        print(f"  -> {model_name}")
        cv_mean, cv_std = cv_auc(model, X_train, y_train, cv_folds)

        # Refit on all training data before touching the held-out test set.
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_prob)

        model_path = (
            f"models/{dataset_name}_{outcome_name}_"
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )
        joblib.dump(model, model_path)

        results.append(
            {
                "Dataset": dataset_name,
                "Outcome": outcome_name,
                "Model": model_name,
                "CV_AUROC_Mean": round(cv_mean, 4),
                "CV_AUROC_SD": round(cv_std, 4),
                "Test_AUROC": round(test_auc, 4),
                "Test_Size": len(y_test),
                "Model_File": model_path,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "knn", "binned"],
        default="all",
    )
    parser.add_argument(
        "--outcome",
        choices=["all", "drop", "threshold"],
        default="all",
    )
    parser.add_argument("--cv", type=int, default=DEFAULT_CV)
    args = parser.parse_args()

    datasets = ["knn", "binned"] if args.dataset == "all" else [args.dataset]
    outcomes = ["drop", "threshold"] if args.outcome == "all" else [args.outcome]

    all_results = []
    for ds in datasets:
        for outcome in outcomes:
            print(f"\n=== {ds.upper()} | {outcome.upper()} ===")
            all_results.extend(run_one(ds, outcome, args.cv))

    df = pd.DataFrame(all_results)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/advanced_results.csv", index=False)

    print("\nAdvanced-model results saved to outputs/advanced_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
