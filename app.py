from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from qml_guard.data import make_demo_data, load_csv, prepare_dataset
from qml_guard.evaluation import evaluate_classifier
from qml_guard.models import classical_models, quantum_qsvc

st.set_page_config(page_title="QML-Guard | Quantum Anomaly Lab", page_icon="⚛️", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
.hero { padding: 1.5rem 1.7rem; border-radius: 18px;
background: linear-gradient(120deg, rgba(13,35,70,.98), rgba(9,92,112,.9));
border: 1px solid rgba(102,210,230,.32); margin-bottom: 1.2rem; }
.hero h1 { margin: 0; color: #f4fbff; font-size: 2.25rem; }
.hero p { margin: .45rem 0 0 0; color: #c9e8f4; font-size: 1.03rem; }
.small-note { color: #94a3b8; font-size: .88rem; }
[data-testid="stMetric"] { background: rgba(100,116,139,.08); padding: 12px; border-radius: 12px; }
</style>
<div class="hero">
  <h1>⚛️ QML-GUARD</h1>
  <p>Hybrid quantum-classical network anomaly detection · Reproducible model evaluation</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment setup")
    data_source = st.radio("Dataset source", ["Synthetic demo", "Upload CSV", "CSV in data/"])
    uploaded = None
    selected_path = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload a labeled CSV", type=["csv"])
    elif data_source == "CSV in data/":
        candidates = sorted(Path("data").glob("*.csv"))
        if candidates:
            selected_path = st.selectbox("Choose dataset", candidates, format_func=lambda p: p.name)
        else:
            st.info("No CSV found in data/. Add a labeled CSV file there.")
    max_samples = st.slider("Maximum rows (quantum run)", 40, 240, 100, 20)
    n_features = st.slider("Selected quantum features", 2, 6, 3)
    reps = st.slider("Quantum circuit repetitions", 1, 3, 2)
    test_size = st.slider("Test-set fraction", 0.15, 0.35, 0.25, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    run = st.button("▶ Run model comparison", type="primary", use_container_width=True)
    st.caption("Quantum-kernel simulation can be computationally expensive. Start small.")

if data_source == "Synthetic demo":
    frame = make_demo_data(n_samples=max(300, max_samples + 20), random_state=int(seed))
elif data_source == "Upload CSV" and uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
elif data_source == "CSV in data/" and selected_path:
    try:
        frame = load_csv(selected_path)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        st.stop()
else:
    frame = None

if frame is None:
    st.info("Choose a dataset source. For a quick first run, select **Synthetic demo**.")
    st.stop()

with st.expander("Preview selected dataset", expanded=False):
    st.write(f"Rows: **{len(frame):,}** · Columns: **{frame.shape[1]}**")
    st.dataframe(frame.head(8), use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None
if run:
    with st.spinner("Preprocessing data and training models. Quantum-kernel runs may take a while..."):
        try:
            X_train, X_test, y_train, y_test, metadata = prepare_dataset(
                frame, test_size=float(test_size), max_samples=int(max_samples),
                n_features=int(n_features), random_state=int(seed)
            )
            results = []
            errors = []
            for name, model in classical_models(int(seed)):
                try:
                    results.append(evaluate_classifier(name, model, X_train, X_test, y_train, y_test))
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            try:
                qmodel = quantum_qsvc(X_train.shape[1], reps=int(reps))
                results.append(evaluate_classifier("Quantum QSVC", qmodel, X_train, X_test, y_train, y_test))
            except Exception as exc:
                errors.append(f"Quantum QSVC: {type(exc).__name__}: {exc}")
            st.session_state.results = {"results": results, "metadata": metadata, "errors": errors,
                                        "y_test": y_test.tolist(), "settings": {
                                            "max_samples": int(max_samples), "features": int(n_features),
                                            "reps": int(reps), "seed": int(seed), "test_size": float(test_size)}}
        except Exception as exc:
            st.error(f"Experiment could not run: {type(exc).__name__}: {exc}")
            st.stop()

payload = st.session_state.results
if payload is None:
    st.markdown("### Experiment workspace")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model families", "Quantum + Classical")
    c2.metric("Evaluation", "Same train/test split")
    c3.metric("Output", "Metrics + confusion matrices")
    st.markdown("""
    **Workflow**
    1. Select the synthetic demo or upload a labeled CSV.
    2. Keep the first run small (80–100 rows, 2–3 features).
    3. Click **Run model comparison**.
    4. Review all metrics and the limitations before presenting results.
    """)
    st.warning("Synthetic demo results validate the pipeline only; use a real labeled network dataset for final research claims.")
    st.stop()

metadata = payload["metadata"]
results = payload["results"]
if not results:
    st.error("No model produced a result. Review the error messages below.")
    st.stop()

st.markdown("### Dataset & experiment overview")
a, b, c, d = st.columns(4)
a.metric("Rows evaluated", f"{metadata['rows_used']:,}")
b.metric("Training rows", f"{metadata['train_rows']:,}")
c.metric("Test rows", f"{metadata['test_rows']:,}")
d.metric("Quantum features", str(len(metadata["selected_features"])))
st.caption(f"Label column: {metadata['label_column']} · Selected features: {', '.join(metadata['selected_features'])}")

df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"confusion_matrix", "predictions"}} for r in results])
st.markdown("### Model comparison")
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
display_df = df[["model"] + metric_cols + ["train_seconds", "predict_seconds"]].copy()
display_df = display_df.rename(columns={
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "roc_auc": "ROC-AUC", "train_seconds": "Train (s)",
    "predict_seconds": "Predict (s)"
})
st.dataframe(display_df.style.format({
    "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "Train (s)": "{:.3f}",
    "Predict (s)": "{:.3f}"
}), use_container_width=True, hide_index=True)

left, right = st.columns([1.15, 1])
with left:
    st.markdown("#### F1-score by model")
    chart_data = df.set_index("model")[["f1"]].rename(columns={"f1": "F1-score"})
    st.bar_chart(chart_data, use_container_width=True)
with right:
    st.markdown("#### Training time")
    st.bar_chart(df.set_index("model")[["train_seconds"]].rename(columns={"train_seconds": "Seconds"}), use_container_width=True)

st.markdown("### Confusion matrices")
cols = st.columns(min(3, len(results)))
for i, result in enumerate(results):
    with cols[i % len(cols)]:
        st.markdown(f"**{result['model']}**")
        cm = result["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Normal", "Attack"])
        ax.set_yticks([0, 1], labels=["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row][col]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

if payload["errors"]:
    with st.expander("Model warnings / errors"):
        for error in payload["errors"]:
            st.warning(error)

st.markdown("### Interpretation")
best = max(results, key=lambda r: r["f1"])
st.write(f"Highest F1-score in this run: **{best['model']}** ({best['f1']:.3f}).")
quantum = next((r for r in results if r["model"] == "Quantum QSVC"), None)
classical_best = max((r for r in results if r["model"] != "Quantum QSVC"), key=lambda r: r["f1"], default=None)
if quantum and classical_best:
    delta = quantum["f1"] - classical_best["f1"]
    st.write(f"Quantum QSVC F1-score minus best classical baseline: **{delta:+.3f}**.")
    if delta > 0:
        st.info("QSVC had a higher F1-score in this run. Repeat with multiple seeds and a real dataset before drawing a conclusion.")
    elif delta < 0:
        st.info("The best classical baseline performed better in this run. This is a valid experimental result; discuss accuracy, cost, and limitations honestly.")
    else:
        st.info("The quantum model and best classical baseline had the same F1-score in this run.")
st.caption("This is an educational research prototype, not a production intrusion prevention system. Performance on synthetic data is not evidence of real-world performance.")
