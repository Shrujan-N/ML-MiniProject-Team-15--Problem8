import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import os

def evaluate_baseline_models():
    print("Evaluating Baseline Models (LASSO, Ridge, ElasticNet, KNN)...")
    
    X_knn = pd.read_csv('data/X_knn_933.csv').values
    X_binned = pd.read_csv('data/X_binned_1367.csv').values
    y_targets = pd.read_csv('data/y_targets.csv')
    
    y_drop = y_targets['y_drop'].values
    y_threshold = y_targets['y_threshold'].values
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    results = []
    
    datasets = {
        'KNN Labs (933)': X_knn,
        'Binned Labs (1367)': X_binned
    }
    
    outcomes = {
        '10 point drop in GPH': y_drop,
        'GPH below 40': y_threshold
    }
    
    models = {
        'LASSO': LogisticRegression(penalty='l1', solver='saga', C=0.2, max_iter=200, tol=1e-2, n_jobs=-1, random_state=42),
        'Ridge': LogisticRegression(penalty='l2', solver='lbfgs', C=0.01, max_iter=200, tol=1e-2, n_jobs=-1, random_state=42),
        'Elastic Net': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.2, max_iter=200, tol=1e-2, n_jobs=-1, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=35, weights='uniform', n_jobs=-1)
    }
    
    for ds_name, X in datasets.items():
        for out_name, y in outcomes.items():
            for model_name, model in models.items():
                scores = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_va = X[train_idx], X[val_idx]
                    y_tr, y_va = y[train_idx], y[val_idx]
                    
                    # Standardize features for hyper-fast convergence
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    X_va_scaled = scaler.transform(X_va)
                    
                    model.fit(X_tr_scaled, y_tr)
                    preds = model.predict_proba(X_va_scaled)[:, 1]
                    scores.append(roc_auc_score(y_va, preds))
                    
                mean_auc = np.mean(scores)
                results.append({
                    'Dataset': ds_name,
                    'Outcome': out_name,
                    'Model': model_name,
                    'AUROC': round(mean_auc, 4)
                })
                print(f"[{ds_name:18s}] | [{out_name:20s}] | {model_name:12s} Mean AUROC: {mean_auc:.4f}")
                
    df_res = pd.DataFrame(results)
    os.makedirs('outputs', exist_ok=True)
    df_res.to_csv('outputs/baseline_results.csv', index=False)
    print("\n✓ Baseline training completed! Saved outputs/baseline_results.csv")

if __name__ == "__main__":
    evaluate_baseline_models()
