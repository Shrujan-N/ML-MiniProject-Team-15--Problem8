"""
Feature ablation analysis + ROC plot for Problem 8.

The current featurization preserves feature prefixes:
demo/util/vital/proc/diag/med/lab. This script uses those prefixes
to remove one feature group at a time, retrains the ensemble, and
records the test AUROC.

It also creates:
    outputs/roc_curves.png
    outputs/ablation_results.csv

Usage:
    python ablation_and_plots.py
    python ablation_and_plots.py --dataset knn --outcome drop
"""

import argparse
import os
import warnings

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
CV_FOLDS = 3


def load_data(dataset):
    path = "data/X_knn_933.csv" if dataset == "knn" else "data/X_binned_1367.csv"
    X = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y_df = pd.read_csv("data/y_targets.csv")
    return X, y_df


def make_models():
    return {
        "LASSO": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1", solver="saga", C=0.2,
                        max_iter=500, random_state=RANDOM_STATE
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
                        penalty="l2", solver="lbfgs", C=0.01,
                        max_iter=500, random_state=RANDOM_STATE
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
                        penalty="elasticnet", solver="saga", l1_ratio=0.5,
                        C=0.2, max_iter=500, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=1.0, kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE
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
        scores.append(
            roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1])
        )
    return float(np.mean(scores))


def get_feature_group_columns(columns):
    groups = {
        "Demographics": ["gender", "insurance_", "race_", "age"],
        "Utilization + baseline": [
            "ed_visits", "hosp_admissions", "hosp_length_stay",
            "ed_to_hosp_ratio", "psych_visits", "baseline_gph"
        ],
        "Vitals": ["vital_"],
        "Procedures": ["proc_"],
        "Diagnoses": ["diag_"],
        "Medications": ["med_"],
        "Labs": ["lab_"],
    }

    mapping = {}
    for group, prefixes in groups.items():
        mapping[group] = [
            c for c in columns if any(c == p or c.startswith(p) for p in prefixes)
        ]
    return mapping


def select_constituents(X_train, y_train):
    """Select the same ensemble constituents once from the full training set."""
    models = make_models()
    scores = {name: cv_score(model, X_train, y_train) for name, model in models.items()}

    logistic = sorted(
        ["LASSO", "Ridge", "Elastic Net"], key=scores.get, reverse=True
    )
    trees = sorted(
        ["Random Forest", "LightGBM"], key=scores.get, reverse=True
    )
    # For the ablation run, keep one best model from each requested family.
    # This keeps the one-week project computationally practical while still
    # implementing the requested logistic + tree + SVM voting ensemble.
    selected = [logistic[0], trees[0], "SVM"]
    return selected, scores


def ensemble_predictions(X_train, y_train, X_test, y_test, selected):
    """Fit the fixed ensemble constituents and average their probabilities."""
    models = make_models()
    probs = []

    for name in selected:
        model = models[name]
        model.fit(X_train, y_train)
        probs.append(model.predict_proba(X_test)[:, 1])

    ensemble_prob = np.mean(probs, axis=0)
    auc = roc_auc_score(y_test, ensemble_prob)
    return auc, ensemble_prob

def run_ablation(dataset, outcome):
    X, y_df = load_data(dataset)
    target = "y_drop" if outcome == "drop" else "y_threshold"
    y = y_df[target].astype(int).to_numpy()

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    groups = get_feature_group_columns(X.columns)
    groups = {k: v for k, v in groups.items() if v}

    os.makedirs("outputs", exist_ok=True)

    # Select the best constituent models ONCE using the full training data.
    # This mirrors the reference paper's approach of selecting a superior
    # ensemble and then ablating feature groups from that model.
    selected, selection_scores = select_constituents(
        X_train_full, y_train
    )

    rows = []
    roc_data = []

    # Full model.
    auc, prob = ensemble_predictions(
        X_train_full, y_train, X_test_full, y_test, selected
    )
    rows.append(
        {
            "Feature_Set": "Full model",
            "Removed_Group": "None",
            "N_Features": X.shape[1],
            "Test_AUROC": round(auc, 4),
            "Selected_Models": ", ".join(selected),
        }
    )
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_data.append(("Full model", fpr, tpr, auc))

    # Ablate one feature group at a time while keeping the selected ensemble
    # fixed. This is substantially faster and is closer to the reference
    # project's ablation experiment.
    for group, cols_to_remove in groups.items():
        keep_cols = [c for c in X.columns if c not in cols_to_remove]
        X_train = X_train_full[keep_cols]
        X_test = X_test_full[keep_cols]

        auc, prob = ensemble_predictions(
            X_train, y_train, X_test, y_test, selected
        )
        rows.append(
            {
                "Feature_Set": f"Without {group}",
                "Removed_Group": group,
                "N_Features": len(keep_cols),
                "Test_AUROC": round(auc, 4),
                "Selected_Models": ", ".join(selected),
            }
        )

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_data.append((f"Without {group}", fpr, tpr, auc))

    results = pd.DataFrame(rows)
    results["AUROC_Change_vs_Full"] = (
        results["Test_AUROC"] - results.loc[0, "Test_AUROC"]
    ).round(4)
    results.to_csv("outputs/ablation_results.csv", index=False)

    plt.figure(figsize=(9, 7))
    for label, fpr, tpr, auc in roc_data:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Ablation Analysis ({dataset}, {outcome})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close()

    selection_df = pd.DataFrame(
        [{"Model": k, "CV_AUROC": round(v, 4)} for k, v in selection_scores.items()]
    ).sort_values("CV_AUROC", ascending=False)
    selection_df.to_csv(
        "outputs/ablation_model_selection.csv", index=False
    )

    print(f"Selected ensemble: {selected}")
    print("\nAblation results:")
    print(results.to_string(index=False))
    print("\nSaved:")
    print("  outputs/ablation_results.csv")
    print("  outputs/ablation_model_selection.csv")
    print("  outputs/roc_curves.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["knn", "binned"], default="knn")
    parser.add_argument(
        "--outcome", choices=["drop", "threshold"], default="drop"
    )
    args = parser.parse_args()
    run_ablation(args.dataset, args.outcome)


if __name__ == "__main__":
    main()
