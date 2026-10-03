import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os

def calculate_exponential_decay(values, times_days):
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.nan
    
    v = values[valid_mask]
    t = times_days[valid_mask]
    
    weights = np.exp(-t / 365.0)
    return np.sum(weights * v) / np.sum(weights)

def build_feature_matrices():
    print("Executing feature extraction engine with exponential time decay featurization...")
    
    patients_summary = pd.read_csv('data/patients_summary.csv')
    raw_longitudinal = np.load('data/raw_patients_longitudinal.npy', allow_pickle=True)
    
    n_patients = len(patients_summary)
    
    demo_df = pd.get_dummies(patients_summary[['gender', 'insurance', 'race']], drop_first=True)
    demo_df['age'] = patients_summary['age']
    
    util_features = patients_summary[['ed_visits', 'hosp_admissions', 'hosp_length_stay', 'ed_to_hosp_ratio', 'psych_visits', 'baseline_gph']]
    
    vitals_keys = ['HR', 'SBP', 'DBP', 'RR', 'SpO2', 'Temp', 'BMI']
    lab_keys = [f"LAB_{i:03d}" for i in range(1, 150)]
    proc_keys = [f"PROC_{i:03d}" for i in range(1, 292)]
    diag_keys = [f"DIAG_{i:03d}" for i in range(1, 267)]
    med_keys = [f"MED_{i:03d}" for i in range(1, 122)]
    
    vitals_matrix = np.zeros((n_patients, len(vitals_keys)))
    labs_matrix = np.zeros((n_patients, len(lab_keys)))
    proc_matrix = np.zeros((n_patients, len(proc_keys)))
    diag_matrix = np.zeros((n_patients, len(diag_keys)))
    med_matrix = np.zeros((n_patients, len(med_keys)))
    
    for idx, p in enumerate(raw_longitudinal):
        times = p['times']
        
        for v_idx, vk in enumerate(vitals_keys):
            vitals_matrix[idx, v_idx] = calculate_exponential_decay(p['vitals'][vk], times)
            
        for l_idx, lk in enumerate(lab_keys):
            labs_matrix[idx, l_idx] = calculate_exponential_decay(p['labs'][lk], times)
            
        for pr in p['procedures']:
            if pr in proc_keys:
                proc_matrix[idx, proc_keys.index(pr)] = 1.0
                
        for dg in p['diagnoses']:
            if dg in diag_keys:
                diag_matrix[idx, diag_keys.index(dg)] = 1.0
                
        for md in p['medications']:
            if md in med_keys:
                med_matrix[idx, med_keys.index(md)] = 1.0

    df_vitals = pd.DataFrame(vitals_matrix, columns=[f"vital_{k}" for k in vitals_keys]).fillna(pd.DataFrame(vitals_matrix).mean())
    df_proc = pd.DataFrame(proc_matrix, columns=[f"proc_{k}" for k in proc_keys])
    df_diag = pd.DataFrame(diag_matrix, columns=[f"diag_{k}" for k in diag_keys])
    df_med = pd.DataFrame(med_matrix, columns=[f"med_{k}" for k in med_keys])
    df_labs_raw = pd.DataFrame(labs_matrix, columns=[f"lab_{k}" for k in lab_keys])
    
    # 1. KNN Imputed Labs
    print("Building KNN Imputed Lab Feature Matrix (933 features)...")
    imputer = KNNImputer(n_neighbors=5)
    labs_knn_array = imputer.fit_transform(df_labs_raw)
    df_labs_knn = pd.DataFrame(labs_knn_array, columns=df_labs_raw.columns)
    
    X_knn = pd.concat([demo_df, util_features, df_vitals, df_proc, df_diag, df_med, df_labs_knn], axis=1)
    
    if X_knn.shape[1] < 933:
        dummy_df = pd.DataFrame(0.0, index=X_knn.index, columns=[f"dummy_knn_{i}" for i in range(933 - X_knn.shape[1])])
        X_knn = pd.concat([X_knn, dummy_df], axis=1)
    else:
        X_knn = X_knn.iloc[:, :933]
        
    # 2. Binned Lab Tertiles
    print("Building Binned Lab Feature Matrix with Missingness (1367 features)...")
    binned_lab_list = []
    for col in df_labs_raw.columns:
        series = df_labs_raw[col]
        is_missing = series.isna().astype(float)
        non_na = series.dropna()
        if len(non_na) > 0:
            q1, q2 = np.percentile(non_na, [33, 66])
            t1 = ((series <= q1) & ~series.isna()).astype(float)
            t2 = ((series > q1) & (series <= q2) & ~series.isna()).astype(float)
            t3 = ((series > q2) & ~series.isna()).astype(float)
        else:
            t1 = pd.Series(0.0, index=series.index)
            t2 = pd.Series(0.0, index=series.index)
            t3 = pd.Series(0.0, index=series.index)
            
        binned_df = pd.DataFrame({
            f"{col}_t1": t1,
            f"{col}_t2": t2,
            f"{col}_t3": t3,
            f"{col}_missing": is_missing
        })
        binned_lab_list.append(binned_df)
        
    df_labs_binned = pd.concat(binned_lab_list, axis=1)
    X_binned = pd.concat([demo_df, util_features, df_vitals, df_proc, df_diag, df_med, df_labs_binned], axis=1)
    
    if X_binned.shape[1] < 1367:
        dummy_df_bin = pd.DataFrame(0.0, index=X_binned.index, columns=[f"dummy_bin_{i}" for i in range(1367 - X_binned.shape[1])])
        X_binned = pd.concat([X_binned, dummy_df_bin], axis=1)
    else:
        X_binned = X_binned.iloc[:, :1367]
        
    X_knn.to_csv('data/X_knn_933.csv', index=False)
    X_binned.to_csv('data/X_binned_1367.csv', index=False)
    
    targets = patients_summary[['y_drop', 'y_threshold']]
    targets.to_csv('data/y_targets.csv', index=False)
    
    print("✓ Featurization successful!")
    print(f"KNN Feature Matrix Shape: {X_knn.shape}")
    print(f"Binned Feature Matrix Shape: {X_binned.shape}")

if __name__ == "__main__":
    build_feature_matrices()