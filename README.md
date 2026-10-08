#### Team No. 15 
#### Problem Statement #8: Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy 
#### Section: K 
#### Team Member 1: - 
#### Name: Shrujan N 
#### SRN: PES1UG24CS624 
#### Team Member 2: - 
#### Name: Dhanush S Shekhar 
#### SRN: PES1UG24CS662 


# ML Mini Project — Team 15, Problem 8

## Project

**Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy**

This repository implements the machine-learning workflow described in the provided
reference project, using the project's synthetic STARR-style dataset.

> **Important:** The dataset in this repository is synthetic. It is generated locally
> by `generate_dataset.py`; it is not the Stanford STARR clinical database and does
> not contain real patient records.

## Assignment scope completed by this code

- Dataset generation
- Feature engineering with exponential time-decay
- KNN-imputed laboratory feature matrix
- Binned laboratory feature matrix
- Baseline logistic/KNN models
- Advanced models:
  - LightGBM
  - Random Forest
  - SVM
  - Multi-Layer Perceptron (FFNN)
- Ensemble soft-voting model
- Feature ablation analysis
- ROC-curve generation
- Reproducible outputs and saved models

The live demonstration, final report, and Q&A preparation are intentionally left
for the team to complete later.

## Repository structure

```text
.
├── data/
├── models/
├── outputs/
├── generate_dataset.py
├── featurization.py
├── train_baselines.py
├── train_advanced.py
├── ensemble.py
├── ablation_and_plots.py
├── requirements.txt
└── README.md
```

## Environment setup

Python 3.10 or 3.11 is recommended.

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Generate the dataset

If the `data/` directory has not already been generated:

```bash
python generate_dataset.py
```

This creates the synthetic patient summary and longitudinal data.

## 2. Build the feature matrices

```bash
python featurization.py
```

This creates:

```text
data/X_knn_933.csv
data/X_binned_1367.csv
data/y_targets.csv
```

The feature engineering includes exponential time decay:

```text
w_i = exp(-t_i / 365)
```

and creates the KNN-imputed and binned laboratory representations.

## 3. Run baseline models

```bash
python train_baselines.py
```

Output:

```text
outputs/baseline_results.csv
```

## 4. Train advanced models

Run all datasets and outcomes:

```bash
python train_advanced.py
```

For a faster/specific run:

```bash
python train_advanced.py --dataset knn --outcome drop
```

Available values:

```text
--dataset: knn | binned | all
--outcome: drop | threshold | all
```

Outputs:

```text
outputs/advanced_results.csv
models/*.joblib
```

The script evaluates the final models on a held-out stratified test split and
uses cross-validation only on the training split.

## 5. Train the ensemble

The ensemble follows the reference project's soft-voting idea:

- LASSO
- Ridge
- Elastic Net
- Random Forest
- LightGBM
- SVM

The constituent models are selected using CV AUROC on the training data and their
test-set probabilities are averaged.

Run:

```bash
python ensemble.py
```

Or:

```bash
python ensemble.py --dataset knn --outcome drop
```

Outputs include:

```text
outputs/ensemble_knn_drop.csv
outputs/ensemble_model_selection_knn_drop.csv
outputs/roc_knn_drop.png
models/ensemble_*.joblib
```

### About the 0.7713 AUROC

The reference project reports **0.7713 AUROC** for its best ensemble on its
original clinical dataset. This repository does **not** hard-code that number.
Our implementation calculates AUROC from the actual predictions produced on the
current synthetic dataset.

Therefore, a different AUROC is expected and is the scientifically correct
behavior unless the original dataset and exact experimental conditions are
available.

## 6. Run feature ablation + ROC curves

```bash
python ablation_and_plots.py
```

Or:

```bash
python ablation_and_plots.py --dataset knn --outcome drop
```

The script first selects the ensemble constituents on the training data,
then keeps a fixed three-model ensemble (best logistic + best tree + SVM) while removing one feature group at a time,
retraining and recording the resulting AUROC. This follows the reference
project's "select the best model, then perform ablation" experiment and is
much faster than re-running model selection for every ablation.

Feature groups include:

- Demographics
- Utilization + baseline
- Vitals
- Procedures
- Diagnoses
- Medications
- Labs

Outputs:

```text
outputs/ablation_results.csv
outputs/roc_curves.png
```

## Reproducibility

The scripts use `random_state=42` wherever supported. The final evaluation uses
a stratified 80/20 train/test split. Model selection uses cross-validation on
the training portion so that the held-out test set is not used for tuning.

## Reference methodology

The project is based on the uploaded reference paper:

**Nicolai Ostberg and Dylan Peterson, "Predicting a Decline in Patient Reported
Outcomes for Cancer Patients on Chemotherapy", CS 229, Spring 2020 Final Project.**

The reference project:
- used EHR-derived features available before chemotherapy,
- created KNN-imputed and binned lab representations,
- compared multiple ML models,
- used AUROC for evaluation,
- built an ensemble voting model,
- and performed feature ablation.

## Current project limitation

The local dataset is a synthetic reconstruction. Its generated targets and
features are not equivalent to the original STARR clinical data. Consequently,
the numerical results should be presented as results of this implementation,
not as a reproduction of the original clinical experiment.

## Suggested Git workflow

```bash
git status
git add train_advanced.py ensemble.py ablation_and_plots.py README.md requirements.txt
git commit -m "Add advanced models ensemble and ablation analysis"
git push origin main
```

Do not commit the virtual environment:

```text
venv/
```

A `.gitignore` should also exclude generated model files and large datasets if
the team does not want them tracked.
