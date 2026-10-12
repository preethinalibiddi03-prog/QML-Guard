# Stage 1 Presentation Outline — QML-Guard

Use actual results from your run. Never replace placeholders with invented metrics.

## Slide 1 — Title
QML-Guard: Hybrid Quantum-Classical Machine Learning for Network Intrusion Detection

## Slide 2 — Problem
- Network traffic may contain normal and malicious patterns.
- A useful detector should identify attacks while limiting false alarms.
- Research question: does a quantum feature map help on a small, controlled benchmark?

## Slide 3 — Proposed solution
- Preprocessing and binary label mapping.
- Qiskit ZZFeatureMap and fidelity quantum kernel.
- QSVC compared with Logistic Regression and RBF SVM.
- Streamlit dashboard for reproducible evaluation.

## Slide 4 — System architecture
Dataset → preprocessing → same train/test split → classical and quantum branches → metrics dashboard.

## Slide 5 — Quantum method
- Encode selected features into a parameterized quantum feature map.
- Estimate similarity with a fidelity quantum kernel.
- Use QSVC to classify the kernel matrix.
- Show the actual feature map circuit from your installed Qiskit version.

## Slide 6 — Experimental setup
Dataset/source, number of rows, label mapping, selected features, split ratio, seed, repetitions, package versions, execution hardware.

## Slide 7 — Results
Insert the dashboard table and confusion matrices from the real dataset. Include accuracy, precision, recall, F1, ROC-AUC where valid, and runtime.

## Slide 8 — Analysis
- Which model had the best F1?
- Which model missed the fewest attacks?
- What was the false-positive trade-off?
- How much slower was quantum simulation?
- Were the results stable across random seeds?

## Slide 9 — Innovation and limitations
- Unified quantum/classical benchmark.
- Transparent metrics and confusion matrices.
- Small sample constraints and simulator cost.
- No claim of quantum advantage unless evidence supports it.

## Slide 10 — Conclusion and future work
Summarize the measured result, then propose repeated-seed evaluation, improved feature selection, noise-aware experiments, and testing on a supported quantum backend.
