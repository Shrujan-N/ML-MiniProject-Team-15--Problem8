import sys
import time

def run_pipeline():
    print("=" * 70)
    print(" Problem 8 Mini-Project Pipeline ")
    print(" Problem #8: Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy ")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Data Generation
    print("\n[STEP 1/5] Running Synthetic STARR EHR Data Generator...")
    import generate_dataset
    generate_dataset.generate_synthetic_starr_data()
    
    # 2. Featurization Engine
    print("\n[STEP 2/5] Running Exponential Time Decay Featurization Engine...")
    import featurization
    featurization.build_feature_matrices()
    
    # 3. Baseline Training
    print("\n[STEP 3/5] Evaluating Baseline Linear & Distance Models...")
    import train_baselines
    train_baselines.evaluate_baseline_models()
    
    # 4. Advanced Models & Ensemble
    print("\n[STEP 4/5] Training Advanced Models (LightGBM, SVM, RF, FFNN)...")
    import train_advanced
    train_advanced.main()
        
    # 5. ROC Plotting & Output Generation
    print("\n[STEP 5/5] Generating ROC Curves & Ablation Results...")
    import ablation_and_plots
    ablation_and_plots.generate_roc_and_ablation()

    print("\n" + "=" * 70)
    print(f"✓ Pipeline execution finished in {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()