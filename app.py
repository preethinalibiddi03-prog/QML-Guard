from pathlib import Path
import time
import base64

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

from qml_guard.data import make_demo_data, load_csv, prepare_dataset
from qml_guard.evaluation import evaluate_classifier
from qml_guard.models import classical_models, quantum_qsvc

st.set_page_config(
    page_title="QML-Guard | Quantum Anomaly Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-tech Styling
st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container { padding-top: 1.2rem; max-width: 1550px; }

.hero-container {
    display: flex;
    align-items: center;
    background: linear-gradient(135deg, rgba(8, 25, 52, 0.95), rgba(12, 74, 96, 0.85));
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 18px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}
.hero-text h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-text p {
    margin: 0.35rem 0 0 0;
    color: #cbd5e1;
    font-size: 1.02rem;
    line-height: 1.4;
}
.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    margin-top: 0.4rem;
}
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.15);
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.upload-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px dashed rgba(56, 189, 248, 0.5);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Hero Header with Visual
banner_path = Path("assets/banner.jpg")
col_hero, col_img = st.columns([2.5, 1.2])
with col_hero:
    st.markdown("""
    <div class="hero-container">
      <div class="hero-text">
        <h1>⚛️ QML-GUARD</h1>
        <p>Hybrid Quantum-Classical Network Intrusion Detection & Reproducible Lab</p>
        <span class="badge-tag">Qiskit 2.x • QSVC Fidelity Kernel • Scikit-learn Baselines</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_img:
    if banner_path.exists():
        st.image(str(banner_path), caption="Quantum Defense Architecture", use_container_width=True)

# Sidebar Experiment Setup
with st.sidebar:
    st.image(str(banner_path), use_container_width=True) if banner_path.exists() else None
    st.header("⚡ Experiment Setup")

    data_source = st.radio(
        "Choose Dataset Source",
        ["NSL-KDD Benchmark (Suitable)", "Upload Custom CSV", "Synthetic Demo", "All CSV in data/"],
        index=0
    )

    uploaded = None
    selected_path = None

    if data_source == "NSL-KDD Benchmark (Suitable)":
        selected_path = Path("data/nsl_kdd_sample.csv")
        st.success("✅ Pre-loaded NSL-KDD intrusion detection dataset with authentic network traffic & attack vectors.")

    elif data_source == "Upload Custom CSV":
        st.markdown("**Upload Network Traffic CSV**")
        uploaded = st.file_uploader(
            "Upload a labeled CSV (NSL-KDD, CIC-IDS, or custom)",
            type=["csv"],
            help="Accepts CSV with normal/attack labels or numeric targets."
        )

        sample_csv_path = Path("data/nsl_kdd_sample.csv")
        if sample_csv_path.exists():
            with open(sample_csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Template / NSL-KDD Sample",
                    data=f.read(),
                    file_name="nsl_kdd_template.csv",
                    mime="text/csv",
                    help="Click to download a ready-to-upload benchmark CSV to test the upload feature!"
                )

    elif data_source == "All CSV in data/":
        candidates = sorted(Path("data").glob("*.csv"))
        if candidates:
            selected_path = st.selectbox("Choose dataset", candidates, format_func=lambda p: p.name)
        else:
            st.info("No CSV found in data/. Add a labeled CSV file there.")

    st.markdown("---")
    st.markdown("##### 🎛️ Quantum & Model Parameters")
    max_samples = st.slider("Maximum rows (quantum run)", 40, 240, 100, 20)
    n_features = st.slider("Selected quantum features (qubits)", 2, 6, 3)
    reps = st.slider("Quantum circuit repetitions", 1, 3, 2)
    test_size = st.slider("Test-set fraction", 0.15, 0.35, 0.25, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    run = st.button("▶ Run Model Comparison", type="primary", use_container_width=True)
    st.caption("ℹ️ Quantum kernel matrix calculation is exact statevector simulation. Recommended rows: 80–120.")

# Dataset Loading
frame = None
if data_source == "NSL-KDD Benchmark (Suitable)" and selected_path and selected_path.exists():
    try:
        frame = load_csv(selected_path)
    except Exception as exc:
        st.error(f"Error loading NSL-KDD sample: {exc}")
elif data_source == "Upload Custom CSV":
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            st.success(f"🎉 Successfully uploaded `{uploaded.name}` ({len(frame):,} rows, {frame.shape[1]} columns)")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            st.stop()
    else:
        st.info("📁 Please drag & drop or upload a labeled CSV file using the uploader in the sidebar, or download our sample template!")
elif data_source == "Synthetic Demo":
    frame = make_demo_data(n_samples=max(300, max_samples + 20), random_state=int(seed))
elif data_source == "All CSV in data/" and selected_path:
    try:
        frame = load_csv(selected_path)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        st.stop()

if frame is None:
    st.stop()

# Interactive Preview & Dataset Overview
with st.expander("📊 Dataset Inspector & Data Profiling", expanded=True):
    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    col_d1.metric("Total Records", f"{len(frame):,}")
    col_d2.metric("Total Features", f"{frame.shape[1] - 1}")
    
    # Identify label col
    label_candidates = [c for c in frame.columns if str(c).lower() in {"label", "class", "target", "outcome", "attack_type"}]
    detected_label = label_candidates[0] if label_candidates else frame.columns[-1]
    col_d3.metric("Detected Label Column", str(detected_label))

    st.dataframe(frame.head(10), use_container_width=True)

    # Class balance preview
    if detected_label in frame.columns:
        counts = frame[detected_label].value_counts()
        st.caption("Class Breakdown:")
        col_bars, col_details = st.columns([2, 1])
        with col_bars:
            st.bar_chart(counts)
        with col_details:
            st.write(counts)

if "results" not in st.session_state:
    st.session_state.results = None

# Running the experiment
if run:
    with st.spinner("⚛️ Preprocessing traffic data & executing Quantum QSVC + Classical baselines..."):
        try:
            X_train, X_test, y_train, y_test, metadata = prepare_dataset(
                frame,
                test_size=float(test_size),
                max_samples=int(max_samples),
                n_features=int(n_features),
                random_state=int(seed),
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

            st.session_state.results = {
                "results": results,
                "metadata": metadata,
                "errors": errors,
                "y_test": y_test.tolist(),
                "settings": {
                    "max_samples": int(max_samples),
                    "features": int(n_features),
                    "reps": int(reps),
                    "seed": int(seed),
                    "test_size": float(test_size),
                },
            }
        except Exception as exc:
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
            st.stop()

payload = st.session_state.results
if payload is None:
    st.markdown("### 🧪 Experiment Workspace")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Families", "Quantum QSVC + Classical")
    c2.metric("Validation Protocol", "Stratified Fair Train/Test Split")
    c3.metric("Output Analytics", "Accuracy, Precision, Recall, F1, ROC-AUC")
    st.info("👈 Configure parameters on the left and click **▶ Run Model Comparison** to train and evaluate.")
    st.stop()

metadata = payload["metadata"]
results = payload["results"]

if not results:
    st.error("No model produced a result. Review error messages.")
    st.stop()

# Overview Metrics
st.markdown("### 📈 Experiment & Feature Overview")
a, b, c, d = st.columns(4)
a.metric("Rows Evaluated", f"{metadata['rows_used']:,}")
b.metric("Training Samples", f"{metadata['train_rows']:,}")
c.metric("Test Samples", f"{metadata['test_rows']:,}")
d.metric("Quantum Qubits / Features", str(len(metadata["selected_features"])))

st.markdown(f"**Selected Quantum Features:** `{', '.join(metadata['selected_features'])}`")

# Results Table
df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"confusion_matrix", "predictions"}} for r in results])
st.markdown("### 🏆 Benchmark Comparison Results")
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
display_df = df[["model"] + metric_cols + ["train_seconds", "predict_seconds"]].copy()
display_df = display_df.rename(columns={
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "roc_auc": "ROC-AUC",
    "train_seconds": "Train Time (s)",
    "predict_seconds": "Inference Time (s)",
})

st.dataframe(
    display_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "Train Time (s)": "{:.3f}",
        "Inference Time (s)": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Comparison Charts
left, right = st.columns([1.1, 1])
with left:
    st.markdown("#### F1-Score by Model")
    chart_data = df.set_index("model")[["f1"]].rename(columns={"f1": "F1-score"})
    st.bar_chart(chart_data, use_container_width=True)
with right:
    st.markdown("#### Training Compute Time (Seconds)")
    st.bar_chart(df.set_index("model")[["train_seconds"]].rename(columns={"train_seconds": "Seconds"}), use_container_width=True)

# Confusion Matrices
st.markdown("### 🎯 Confusion Matrices")
cols = st.columns(min(3, len(results)))
for i, result in enumerate(results):
    with cols[i % len(cols)]:
        st.markdown(f"**{result['model']}**")
        cm = result["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Normal", "Attack"])
        ax.set_yticks([0, 1], labels=["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row][col]), ha="center", va="center", color="black" if cm[row][col] < np.max(cm)/2 else "white")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# Quantum Feature Map Visualization
with st.expander("⚛️ Quantum Circuit Architecture (ZZFeatureMap)", expanded=False):
    st.markdown("""
    The quantum model encodes the selected classical network traffic features into an \(n\)-qubit Hilbert space using a **ZZFeatureMap** with linear entanglement:
    """)
    try:
        from qiskit.circuit.library import ZZFeatureMap
        n_q = len(metadata["selected_features"])
        fmap = ZZFeatureMap(feature_dimension=n_q, reps=int(payload["settings"]["reps"]), entanglement="linear")
        circuit_str = str(fmap.decompose().draw(output="text"))
        st.code(circuit_str, language="text")
    except Exception as exc:
        st.caption(f"Circuit visualization: {exc}")

if payload["errors"]:
    with st.expander("⚠️ Model Warnings / Errors"):
        for error in payload["errors"]:
            st.warning(error)

# Scientific Conclusion
st.markdown("### 🔬 Experimental Findings & Honesty")
best = max(results, key=lambda r: r["f1"])
st.write(f"Highest overall F1-score: **{best['model']}** ({best['f1']:.3f}).")
quantum = next((r for r in results if r["model"] == "Quantum QSVC"), None)
classical_best = max((r for r in results if r["model"] != "Quantum QSVC"), key=lambda r: r["f1"], default=None)

if quantum and classical_best:
    delta = quantum["f1"] - classical_best["f1"]
    st.write(f"Quantum QSVC vs. Best Classical Baseline ΔF1: **{delta:+.3f}**.")
    if delta > 0:
        st.success("Quantum QSVC achieved a higher F1-score on this feature subset. Evaluate with multiple seeds to confirm statistical significance.")
    elif delta < 0:
        st.info("The classical baseline performed higher on this split. This demonstrates honest empirical research without bias.")
    else:
        st.info("Both models achieved identical F1 scores on this test split.")

st.caption("Educational research prototype developed for reproducible quantum-classical benchmarking.")
