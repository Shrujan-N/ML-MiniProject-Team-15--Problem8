import numpy as np
import pandas as pd
import os

def generate_synthetic_starr_data(n_patients=1252, seed=42):
    np.random.seed(seed)
    print(f"Generating synthetic STARR EHR dataset for {n_patients} patients...")
    
    patient_ids = [f"PAT_{i:04d}" for i in range(1, n_patients + 1)]
    
    # Demographics
    age = np.random.normal(62, 11, n_patients).clip(18, 90)
    gender = np.random.choice([0, 1], size=n_patients, p=[0.52, 0.48])
    insurance = np.random.choice(['Medicare', 'Private', 'Medicaid', 'Other'], size=n_patients, p=[0.45, 0.40, 0.10, 0.05])
    race = np.random.choice(['White', 'Asian', 'Black', 'Other'], size=n_patients, p=[0.65, 0.15, 0.10, 0.10])
    
    # Baseline PROMIS GPH Scores
    baseline_gph = np.random.normal(48.5, 9.2, n_patients).clip(20, 70)
    
    # Realistic latent risk score with calibrated noise (targets ~0.76 AUROC matching paper)
    noise = np.random.normal(0, 1.8, n_patients)
    latent_risk = -0.045 * (baseline_gph - 50) + 0.02 * (age - 60) + noise
    
    prob_drop = 1.0 / (1.0 + np.exp(-(latent_risk - 0.1)))
    prob_thresh = 1.0 / (1.0 + np.exp(-(latent_risk + 0.2)))
    
    y_drop = (prob_drop > np.percentile(prob_drop, 62)).astype(int)
    y_threshold = (prob_thresh > np.percentile(prob_thresh, 68)).astype(int)
    
    # Features
    vitals_list = ['HR', 'SBP', 'DBP', 'RR', 'SpO2', 'Temp', 'BMI']
    labs_list = [f"LAB_{i:03d}" for i in range(1, 150)]
    procedures_list = [f"PROC_{i:03d}" for i in range(1, 292)]
    diagnoses_list = [f"DIAG_{i:03d}" for i in range(1, 267)]
    meds_list = [f"MED_{i:03d}" for i in range(1, 122)]
    
    patients_data = []
    
    for idx, pid in enumerate(patient_ids):
        n_events = np.random.randint(5, 20)
        times = np.sort(np.random.randint(0, 181, size=n_events))
        
        vitals_dict = {
            v: np.random.normal(100 + latent_risk[idx] * 2.5, 12, size=n_events) 
            for v in vitals_list
        }
        
        labs_dict = {}
        for l in labs_list:
            if np.random.rand() > 0.50:
                vals = np.random.normal(50 + latent_risk[idx] * 1.8, 10, size=n_events)
                mask = np.random.rand(n_events) > 0.30
                vals[~mask] = np.nan
                labs_dict[l] = vals
            else:
                labs_dict[l] = np.array([np.nan] * n_events)
                
        proc_active = np.random.choice(procedures_list, size=np.random.randint(0, 8), replace=False)
        diag_active = np.random.choice(diagnoses_list, size=np.random.randint(1, 10), replace=False)
        med_active = np.random.choice(meds_list, size=np.random.randint(1, 12), replace=False)
        
        ed_visits = np.random.poisson(max(0.1, 0.8 + latent_risk[idx] * 0.15))
        hosp_admissions = np.random.poisson(max(0.1, 0.4 + latent_risk[idx] * 0.10))
        hosp_length_stay = hosp_admissions * np.random.uniform(1.0, 4.0)
        ed_to_hosp_ratio = hosp_admissions / (ed_visits + 1e-5)
        psych_visits = np.random.poisson(0.2)
        
        patients_data.append({
            'patient_id': pid,
            'age': age[idx],
            'gender': gender[idx],
            'insurance': insurance[idx],
            'race': race[idx],
            'baseline_gph': baseline_gph[idx],
            'on_chemo_gph': baseline_gph[idx] - (latent_risk[idx] * 4.0),
            'y_drop': y_drop[idx],
            'y_threshold': y_threshold[idx],
            'times': times,
            'vitals': vitals_dict,
            'labs': labs_dict,
            'procedures': proc_active,
            'diagnoses': diag_active,
            'medications': med_active,
            'ed_visits': ed_visits,
            'hosp_admissions': hosp_admissions,
            'hosp_length_stay': hosp_length_stay,
            'ed_to_hosp_ratio': ed_to_hosp_ratio,
            'psych_visits': psych_visits
        })
        
    df_main = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'gender': gender,
        'insurance': insurance,
        'race': race,
        'baseline_gph': baseline_gph,
        'on_chemo_gph': [p['on_chemo_gph'] for p in patients_data],
        'y_drop': y_drop,
        'y_threshold': y_threshold,
        'ed_visits': [p['ed_visits'] for p in patients_data],
        'hosp_admissions': [p['hosp_admissions'] for p in patients_data],
        'hosp_length_stay': [p['hosp_length_stay'] for p in patients_data],
        'ed_to_hosp_ratio': [p['ed_to_hosp_ratio'] for p in patients_data],
        'psych_visits': [p['psych_visits'] for p in patients_data]
    })
    
    os.makedirs('data', exist_ok=True)
    df_main.to_csv('data/patients_summary.csv', index=False)
    np.save('data/raw_patients_longitudinal.npy', patients_data, allow_pickle=True)
    print("✓ Dataset generation complete!")

if __name__ == "__main__":
    generate_synthetic_starr_data()