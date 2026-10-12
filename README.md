# QML-Guard
### Hybrid Quantum-Classical Network Intrusion Detection

QML-Guard is a hackathon-ready research prototype that compares a Qiskit quantum-kernel classifier (QSVC) against classical baselines for binary network-traffic classification: **normal vs. attack**.

> Research honesty: this project is designed to test whether a quantum feature map helps on a selected, reduced dataset. It does not assume quantum advantage. Results depend on the dataset, feature selection, machine, and simulation settings.

## Features
- Dataset loader for NSL-KDD-style CSV files and a small synthetic demo mode.
- Reproducible train/test split and preprocessing pipeline.
- Classical baselines: Logistic Regression and RBF SVM.
- Quantum model: Qiskit `QSVC` with a `FidelityQuantumKernel`.
- Metrics: accuracy, precision, recall, F1, ROC-AUC where available, confusion matrix, and training/prediction time.
- Streamlit dashboard for demo, training, comparison, and prediction.
- CPU-friendly controls: small sample size and feature count because quantum-kernel simulation scales quickly.

## Architecture
```text
CSV / Demo data
      |
Validation + preprocessing (scikit-learn)
      |
Stratified train/test split
      |
      +-------------------------+
      |                         |
Classical baselines       Qiskit quantum kernel
Logistic Regression       ZZFeatureMap -> FidelityQuantumKernel -> QSVC
RBF SVM                         |
      +------------+------------+
                   |
      Metrics + confusion matrices + dashboard
```

## Quick start (Windows PowerShell)
Use Python 3.11 or 3.12 for the smoothest compatibility with the quantum stack.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Demo without downloading a dataset
1. Start the app.
2. Select **Synthetic demo** in the sidebar.
3. Click **Run comparison**.
4. Use a small sample (start at 80–120 rows) and 2–4 features.

Synthetic data is only for validating the workflow and UI; do not present it as evidence of real-world intrusion detection.

## Use NSL-KDD data
Download the NSL-KDD dataset from the official Canadian Institute for Cybersecurity dataset page:
https://www.unb.ca/cic/datasets/nsl.html

Place a CSV file in `data/` (the folder is included with `.gitkeep`) and choose it in the app. The loader supports:
- CSV with a `label` or `class` column; or
- NSL-KDD-style rows where the last column is the label.

For NSL-KDD, labels such as `normal` are treated as normal; other labels are treated as attacks. Confirm the column layout and remove header rows if your downloaded file does not contain column names.

## How the experiment works
1. Loads a dataset and maps labels to `0 = normal`, `1 = attack`.
2. Samples a manageable subset, stratified by label.
3. Keeps numeric features and one-hot encodes categorical columns.
4. Imputes missing values, scales features, and selects a small number of features.
5. Trains classical baselines and a QSVC quantum-kernel model on the same split.
6. Reports metrics and elapsed time.

For fair comparisons, use the same dataset split for all models. Record the sample size, random seed, selected features, package versions, and simulator/hardware backend in your presentation.

## Project structure
```text
QML-Guard/
├── app.py
├── qml_guard/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── evaluation.py
├── tests/
│   └── test_data.py
├── data/
│   └── .gitkeep
├── docs/
│   ├── EXPERIMENT_PLAN.md
│   └── PRESENTATION_OUTLINE.md
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Troubleshooting
- **Qiskit install errors:** use Python 3.11/3.12 in a fresh virtual environment and reinstall `requirements.txt`.
- **Slow quantum run:** reduce sample size to 60–100, use 2–3 features, and start with the demo.
- **No quantum result:** the app reports the exception rather than fabricating a score. Classical models can still run.
- **Dataset columns:** ensure the label column is correctly identified; inspect the CSV before drawing conclusions.

## Research limitations
- Exact quantum-kernel simulation may be slow as sample count grows.
- One-hot encoding and feature selection can affect results.
- A small benchmark is a prototype, not proof of production-level detection.
- Never use this demo alone to block live traffic or make security decisions.
