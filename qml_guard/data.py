from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


COMMON_LABELS = {"label", "class", "target", "outcome", "attack_type"}


def make_demo_data(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """Create clearly labeled synthetic data for testing the pipeline/UI only."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.72, 0.28],
        class_sep=0.9,
        random_state=random_state,
    )
    frame = pd.DataFrame(X, columns=[f"traffic_feature_{i+1}" for i in range(X.shape[1])])
    frame["label"] = np.where(y == 0, "normal", "attack")
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("The CSV must contain at least one feature column and one label column.")
    return frame


def _find_label_column(frame: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in COMMON_LABELS:
        if candidate in normalized:
            return normalized[candidate]
    # NSL-KDD-style datasets often have the target in the final column.
    return frame.columns[-1]


def _to_binary_labels(values: pd.Series) -> np.ndarray:
    labels = values.astype(str).str.strip().str.lower()
    # Handles common normal labels and labels such as "normal.".
    normal = labels.isin({"normal", "normal.", "benign", "0", "false"})
    # If labels are already binary, preserve the 0/1 meaning.
    unique = set(labels.unique())
    if unique.issubset({"0", "1"}):
        return labels.astype(int).to_numpy()
    y = (~normal).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("The chosen label column has only one class after mapping. Check the CSV label column.")
    return y


def prepare_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    max_samples: int = 180,
    n_features: int = 4,
    random_state: int = 42,
):
    """Return transformed train/test arrays, labels, selected feature names and metadata."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    label_col = _find_label_column(frame)
    clean = frame.dropna(subset=[label_col]).copy()
    y = _to_binary_labels(clean[label_col])
    X = clean.drop(columns=[label_col]).copy()

    # Drop likely identifier columns and columns with only one value.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.drop(columns=[c for c in X.columns if str(c).strip().lower() in {"id", "index", "flow_id"}], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must include both normal and attack examples.")

    if len(X) > max_samples:
        _, X, _, y = train_test_split(
            X, y, test_size=max_samples, random_state=random_state, stratify=y
        )

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    X_train = np.nan_to_num(np.asarray(X_train, dtype=float))
    X_test = np.nan_to_num(np.asarray(X_test, dtype=float))

    try:
        all_names = preprocessor.get_feature_names_out()
    except Exception:
        all_names = np.array([f"feature_{i+1}" for i in range(X_train.shape[1])])

    k = min(max(1, int(n_features)), X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    selected_names = np.asarray(all_names)[selector.get_support()].tolist()

    # Quantum feature maps expect bounded inputs. Map standardized values into [0, pi].
    # Fit bounds using training data only to avoid test-set leakage.
    low = np.nanmin(X_train, axis=0)
    high = np.nanmax(X_train, axis=0)
    span = np.where((high - low) < 1e-9, 1.0, high - low)
    X_train = np.clip((X_train - low) / span, 0, 1) * np.pi
    X_test = np.clip((X_test - low) / span, 0, 1) * np.pi

    metadata = {
        "label_column": str(label_col),
        "rows_used": int(len(X)),
        "original_features": int(frame.shape[1] - 1),
        "encoded_features": int(len(all_names)),
        "selected_features": selected_names,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "normal_count": int(np.sum(y == 0)),
        "attack_count": int(np.sum(y == 1)),
    }
    return X_train, X_test, y_train, y_test, metadata
