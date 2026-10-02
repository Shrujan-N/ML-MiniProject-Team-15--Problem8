"""
Ensemble Voting Model for Problem 8.

This follows the reference project's ensemble idea:
- Logistic models: LASSO, Ridge, Elastic Net
- Tree models: Random Forest, LightGBM
- SVM
- average predicted probabilities (soft voting)

The "top" constituents are selected using CV AUROC on the training split.
The held-out test split is used only for the final evaluation.

Usage:
    python ensemble.py
    python ensemble.py --dataset knn --outcome drop
"""

import argparse
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def load_data(dataset_name, outcome_name):
    path = (
        "data/X_knn_933.csv"
        if dataset_name == "knn"
        else "data/X_binned_1367.csv"
    )
    X = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    target = "y_drop" if outcome_name == "drop" else "y_threshold"
    y = pd.read_csv("data/y_targets.csv")[target].astype(int).to_numpy()
    return X, y


def make_candidates():
    return {
        "LASSO": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        C=0.2,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.01,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        C=0.2,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=31,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            verbosity=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
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
    }


def cv_score(model, X, y):
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = []

    for tr, va in skf.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y[va], prob))

    return float(np.mean(scores))


def fit_ensemble(dataset_name, outcome_name):
    X, y = load_data(dataset_name, outcome_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    candidates = make_candidates()
    cv_scores = {}

    print(f"\nSelecting ensemble constituents: {dataset_name} / {outcome_name}")
    for name, model in candidates.items():
        score = cv_score(model, X_train, y_train)
        cv_scores[name] = score
        print(f"{name:15s} CV AUROC = {score:.4f}")

    # Match the structure described in the reference paper:
    # up to 3 logistic models + up to 2 tree models + the best SVM.
    logistic_names = ["LASSO", "Ridge", "Elastic Net"]
    tree_names = ["Random Forest", "LightGBM"]

    selected = sorted(logistic_names, key=cv_scores.get, reverse=True)
    selected += sorted(tree_names, key=cv_scores.get, reverse=True)
    selected.append("SVM")

    # Keep the six constituent models, as in the reference ensemble.
    selected = list(dict.fromkeys(selected))[:6]

    probabilities = []
    fitted = {}

    for name in selected:
        model = candidates[name]
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        probabilities.append(prob)
        fitted[name] = model

    ensemble_prob = np.mean(probabilities, axis=0)
    ensemble_auc = roc_auc_score(y_test, ensemble_prob)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    result = pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Outcome": outcome_name,
                "Selected_Models": ", ".join(selected),
                "Ensemble_Test_AUROC": round(ensemble_auc, 4),
                "Reference_AUROC": 0.7713,
                "Note": "Reference value; not hard-coded into prediction.",
            }
        ]
    )
    result.to_csv(
        f"outputs/ensemble_{dataset_name}_{outcome_name}.csv",
        index=False,
    )

    pd.DataFrame(
        [{"Model": k, "CV_AUROC": round(v, 4)} for k, v in cv_scores.items()]
    ).sort_values("CV_AUROC", ascending=False).to_csv(
        f"outputs/ensemble_model_selection_{dataset_name}_{outcome_name}.csv",
        index=False,
    )

    for name, model in fitted.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(
            model,
            f"models/ensemble_{dataset_name}_{outcome_name}_{safe}.joblib",
        )

    # ROC curve for the ensemble.
    fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Ensemble (AUROC = {ensemble_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Ensemble ROC: {dataset_name} / {outcome_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"outputs/roc_{dataset_name}_{outcome_name}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSelected models: {selected}")
    print(f"Ensemble test AUROC: {ensemble_auc:.4f}")
    print("Results saved under outputs/ and models/")

    return ensemble_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["knn", "binned"], default="knn")
    parser.add_argument(
        "--outcome", choices=["drop", "threshold"], default="drop"
    )
    args = parser.parse_args()
    fit_ensemble(args.dataset, args.outcome)


if __name__ == "__main__":
    main()
