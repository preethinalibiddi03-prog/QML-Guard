from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = np.asarray(model.predict(X_test)).astype(int)
    predict_seconds = time.perf_counter() - start

    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
        elif hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(auc),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "confusion_matrix": cm.tolist(),
        "predictions": predictions.tolist(),
    }
