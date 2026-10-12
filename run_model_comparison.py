import numpy as np
import pandas as pd
from qml_guard.data import load_csv, prepare_dataset
from qml_guard.models import classical_models, quantum_qsvc
from qml_guard.evaluation import evaluate_classifier

def run_comparison():
    print("=" * 65)
    print("         QML-GUARD: FAST MODEL COMPARISON & EVALUATION")
    print("=" * 65)

    df = load_csv("data/nsl_kdd_sample.csv")
    X_train, X_test, y_train, y_test, meta = prepare_dataset(
        df, max_samples=60, n_features=2, test_size=0.25, random_state=42
    )

    print("\n--- 1. DATASET PROFILE ---")
    print(f"Total Rows Evaluated: {meta['rows_used']}")
    print(f"Training Samples:     {meta['train_rows']}")
    print(f"Testing Samples:      {meta['test_rows']}")
    print(f"Normal Samples:       {meta['normal_count']}")
    print(f"Attack Samples:       {meta['attack_count']}")
    print(f"Selected Features:    {', '.join(meta['selected_features'])}\n")

    results = []
    print("--- 2. TRAINING MODELS ---")
    for name, model in classical_models(42):
        r = evaluate_classifier(name, model, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"  * {name:<20} finished in {r['train_seconds']:.3f}s")

    print("\nTraining Quantum QSVC (ZZFeatureMap + Fidelity Kernel)...")
    qmodel = quantum_qsvc(len(meta["selected_features"]), reps=1)
    qr = evaluate_classifier("Quantum QSVC", qmodel, X_train, X_test, y_train, y_test)
    results.append(qr)
    print(f"  * {'Quantum QSVC':<20} finished in {qr['train_seconds']:.3f}s\n")

    print("--- 3. COMPARATIVE PERFORMANCE METRICS ---")
    df_res = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1-Score": f"{r['f1']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}" if not np.isnan(r["roc_auc"]) else "N/A",
        "Train Time": f"{r['train_seconds']:.3f}s",
        "Predict Time": f"{r['predict_seconds']:.3f}s"
    } for r in results])
    print(df_res.to_string(index=False))

    print("\n--- 4. DETAILED CONFUSION MATRICES ---")
    for r in results:
        cm = np.array(r["confusion_matrix"])
        print(f"\n[{r['model']}]")
        print(f"  True Negative  (Normal Traffic Identified): {cm[0,0]}")
        print(f"  False Positive (False Alarm / False Attack): {cm[0,1]}")
        print(f"  False Negative (Missed Attack):             {cm[1,0]}")
        print(f"  True Positive  (Attack Blocked):            {cm[1,1]}")

    print("\n" + "=" * 65)
    best_f1 = max(results, key=lambda r: r["f1"])
    print(f"Top Model by F1-Score: {best_f1['model']} (F1 = {best_f1['f1']:.4f})")
    print("=" * 65)

if __name__ == "__main__":
    run_comparison()
