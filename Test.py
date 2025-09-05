import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import confusion_matrix_plot

st.title("Test / Demo")

st.session_state.setdefault("models", {})
if not st.session_state["models"]:
    st.info("No models in memory. Train a model or load one in the **Training** or **Deep Learning** page.")
    st.stop()

model_names = list(st.session_state["models"].keys())
default_model_name = st.session_state.get("last_model_name", model_names[0])
default_index = model_names.index(default_model_name) if default_model_name in model_names else 0
pick_model = st.selectbox("Choose a model", model_names, index=default_index)
model = st.session_state["models"][pick_model]
st.session_state["last_model_name"] = pick_model

CANON_LABELS = ("negative", "positive")
LABEL_TO_INT = {CANON_LABELS[0]: 0, CANON_LABELS[1]: 1}
INT_TO_LABEL = {0: CANON_LABELS[0], 1: CANON_LABELS[1]}

def _to_str_labels(y_any):
    s = pd.Series(list(y_any))
    if pd.api.types.is_numeric_dtype(s):
        uniq = set(s.dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return [INT_TO_LABEL[int(v)] for v in s.astype(int).tolist()]
        return [str(v) for v in s.tolist()]
    else:
        vals = s.astype(str).str.lower().tolist()
        mapped = []
        for v in vals:
            if v in LABEL_TO_INT:
                mapped.append(v)
            elif v in {"neg", "pos"}:
                mapped.append("negative" if v == "neg" else "positive")
            elif v in {"0", "1"}:
                mapped.append(INT_TO_LABEL[int(v)])
            else:
                mapped.append(v)
        return mapped

def _pred_to_str_labels(pred_any):
    if not isinstance(pred_any, (list, tuple, np.ndarray, pd.Series)):
        pred_any = [pred_any]
    s = pd.Series(list(pred_any))

    if pd.api.types.is_numeric_dtype(s):
        uniq = set(s.dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return [INT_TO_LABEL[int(v)] for v in s.astype(int).tolist()]
        return [str(v) for v in s.tolist()]
    else:
        vals = s.astype(str).str.lower().tolist()
        mapped = []
        for v in vals:
            if v in LABEL_TO_INT:
                mapped.append(v)
            elif v in {"neg", "pos"}:
                mapped.append("negative" if v == "neg" else "positive")
            elif v in {"0", "1"}:
                mapped.append(INT_TO_LABEL[int(v)])
            else:
                mapped.append(v)
        return mapped

st.markdown("### Single text prediction")
user_text = st.text_area("Enter a text to predict:", value="This movie was great ! amazing !", height=100)
if st.button("Predict"):
    try:
        yhat = model.predict([user_text])
        yhat_str = _pred_to_str_labels(yhat)
        st.success(f"Prediction: **{yhat_str[0]}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.subheader("🧪 Evaluate on a dataset sample")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.info("No datasets in memory. Add some from the **Exploration** page.")
else:
    ds_names = list(st.session_state["datasets"].keys())
    default_ds = st.session_state.get("last_dataset_name", ds_names[0])
    default_ds_idx = ds_names.index(default_ds) if default_ds in ds_names else 0
    ds_choice = st.selectbox("Dataset to sample", ds_names, index=default_ds_idx)
    st.session_state["last_dataset_name"] = ds_choice

    ds = st.session_state["datasets"][ds_choice]
    df = ds["df"].copy()
    label_col = ds["label_col"]

    if "__text__" not in df.columns:
        st.error("Dataset missing '__text__' column.")
    else:
        k = st.slider("Sample size", 5, min(200, len(df)), min(20, len(df)))
        sample = df.sample(k, random_state=42)

        Xs = sample["__text__"].astype(str).tolist()
        ys_true_str = _to_str_labels(sample[label_col].tolist())

        try:
            yhat_raw = model.predict(Xs)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        yhat_str = _pred_to_str_labels(yhat_raw)

        acc = accuracy_score(ys_true_str, yhat_str)
        st.write(f"Sample accuracy: **{acc:.3f}**")

        out = sample.copy()
        out["y_true"] = ys_true_str
        out["y_pred"] = yhat_str
        st.dataframe(out[["__text__", "y_true", "y_pred"]])

        try:
            confusion_matrix_plot(ys_true_str, yhat_str, st)
        except Exception as e:
            st.warning(f"Confusion matrix could not be computed: {e}")
