import pandas as pd
from qml_guard.data import load_csv, prepare_dataset
from qml_guard.models import classical_models, quantum_qsvc
from qml_guard.evaluation import evaluate_classifier

def run_experiment():
    print("============================================================")
    print("   QML-GUARD: HYBRID QUANTUM-CLASSICAL INTRUSION DETECTION  ")
    print("============================================================\n")

    print("Loading NSL-KDD suitable benchmark dataset...")
    df = load_csv("data/nsl_kdd_sample.csv")
    X_train, X_test, y_train, y_test, meta = prepare_dataset(
        df, max_samples=80, n_features=3, random_state=42
    )

    print(f"Dataset Size:       {meta['rows_used']} total records")
    print(f"Training Samples:   {meta['train_rows']}")
    print(f"Test Samples:       {meta['test_rows']}")
    print(f"Selected Features:  {', '.join(meta['selected_features'])}\n")

    results = []
    print("Training Classical Baseline Models...")
    for name, model in classical_models(42):
        r = evaluate_classifier(name, model, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"  * {name:<20} | F1: {r['f1']:.3f} | Train Time: {r['train_seconds']:.3f}s")

    print("\nTraining Quantum QSVC (Qiskit ZZFeatureMap Fidelity Kernel)...")
    qmodel = quantum_qsvc(len(meta["selected_features"]), reps=2)
    qr = evaluate_classifier("Quantum QSVC", qmodel, X_train, X_test, y_train, y_test)
    results.append(qr)
    print(f"  * {'Quantum QSVC':<20} | F1: {qr['f1']:.3f} | Train Time: {qr['train_seconds']:.3f}s\n")

    res_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.3f}",
        "Precision": f"{r['precision']:.3f}",
        "Recall": f"{r['recall']:.3f}",
        "F1-Score": f"{r['f1']:.3f}",
        "Train Time": f"{r['train_seconds']:.3f}s",
        "Inference Time": f"{r['predict_seconds']:.3f}s",
    } for r in results])

    print("======================= FINAL RESULTS =======================")
    print(res_df.to_string(index=False))
    print("=============================================================")

if __name__ == "__main__":
    run_experiment()
