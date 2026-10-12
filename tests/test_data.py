import numpy as np
from qml_guard.data import make_demo_data, prepare_dataset


def test_demo_data_is_binary_and_preparable():
    frame = make_demo_data(n_samples=120, random_state=7)
    X_train, X_test, y_train, y_test, metadata = prepare_dataset(
        frame, max_samples=100, n_features=3, random_state=7
    )
    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})
    assert metadata["train_rows"] + metadata["test_rows"] <= 100
