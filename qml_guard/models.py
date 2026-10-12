from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def classical_models(random_state: int = 42):
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ("Classical RBF SVM", SVC(kernel="rbf", class_weight="balanced")),
    ]


def quantum_qsvc(n_features: int, reps: int = 2):
    """Build a QSVC using Qiskit's statevector-based fidelity quantum kernel."""
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC
    except ImportError as exc:
        raise ImportError(
            "Qiskit ML dependencies are missing. Install packages from requirements.txt "
            "inside a Python 3.11/3.12 virtual environment."
        ) from exc

    feature_map = ZZFeatureMap(feature_dimension=int(n_features), reps=int(reps), entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    return QSVC(quantum_kernel=quantum_kernel)
