# Experiment Plan

## Research question
On a reduced binary network-intrusion classification task, how does a Qiskit quantum-kernel SVM compare with classical Logistic Regression and an RBF SVM?

## Hypothesis
The quantum feature map may represent some feature relationships differently, but it may not outperform classical baselines. We test rather than assume advantage.

## Experimental protocol
1. Choose a labeled network dataset (preferably NSL-KDD for the prototype).
2. State exactly how labels are mapped to normal (0) and attack (1).
3. Use a stratified split, fixed seed, and the same transformed data for all models.
4. Keep quantum training subset small because kernel construction requires many circuit evaluations.
5. Report accuracy, precision, recall, F1-score, ROC-AUC where supported, and runtime.
6. Repeat with at least three seeds for the final report if time permits.
7. Record Python/Qiskit versions, sample counts, feature count, circuit repetitions, and simulator/backend.

## Fairness and leakage checks
- Fit imputation, encoding, scaling, and feature selection on training data only.
- Do not select the best run and present it as the only run.
- Compare all models on the same held-out test split.
- Do not tune hyperparameters on the test set.
- Clearly distinguish synthetic demo results from real dataset results.

## Limitations
- Quantum-kernel simulation can be slow.
- The selected subset may not represent the full dataset.
- Classical models are strong baselines for tabular data.
- Simulation does not establish an advantage on quantum hardware.
