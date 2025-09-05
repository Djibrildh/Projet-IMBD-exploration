import joblib
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from utils import (
    build_estimator, default_param_grid, make_pipeline,
    bytes_downloadable_model, confusion_matrix_plot
)

st.title("Training (TF-IDF + Classical ML)")

st.session_state.setdefault("models", {})
st.session_state.setdefault("last_model_name", None)

if not st.session_state.get("datasets"):
    st.warning("➡ Go to the **Exploration** page first and add at least one dataset.")
    st.stop()

st.subheader("Choose dataset")
ds_names = list(st.session_state["datasets"].keys())
ds_choice = st.selectbox("Dataset", ds_names, index=ds_names.index(st.session_state.get("last_dataset_name", ds_names[0])))
st.session_state["last_dataset_name"] = ds_choice

ds = st.session_state["datasets"][ds_choice]
df = ds["df"]; label_col = ds["label_col"]

X = df["__text__"].astype(str).tolist()
y_series = df[label_col].astype(str).map({"negative":0, "positive":1})
y = y_series.values if not y_series.isna().any() else df[label_col].values

st.subheader("Quick params")
colA, colB, colC = st.columns(3)
with colA:
    model_name = st.selectbox("Model", ["LogisticRegression",
                                        "SVC",
                                        "NaiveBayes (Multinomial)",
                                        "PassiveAggressive",
                                        "RandomForest"])
with colB:
    max_features = st.slider("TF-IDF max_features", 5_000, 50_000, 20_000, step=5_000)
    use_bigrams = st.checkbox("Use bigrams (1,2)", value=True)
with colC:
    test_size = st.slider("Test size (%)", 10, 40, 20, step=5)
    subset = st.number_input("Subsample (0 = all)", min_value=0, max_value=len(X), value=0, step=1000)

if subset and subset < len(X):
    X_small, _, y_small, _ = train_test_split(X, y, train_size=subset, stratify=y, random_state=42)
else:
    X_small, y_small = X, y

X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=test_size/100, stratify=y_small, random_state=42
)

st.subheader("Optimization")
col1, col2 = st.columns(2)
with col1:
    opt_method = st.radio("Method", ["None", "GridSearch"], index=0)
with col2:
    proba_svc = st.checkbox("SVC: enable probabilities (slower)", value=False)

estimator = build_estimator(model_name, proba=proba_svc)
pipe = make_pipeline(estimator, max_features=max_features, use_bigrams=use_bigrams)

if st.button("Train"):
    with st.spinner("Training..."):
        model = pipe
        if opt_method == "None":
            model.fit(X_train, y_train)
        else:
            param_grid = default_param_grid(model_name, proba=proba_svc)
            search = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            st.success(f"Best params: {getattr(search, 'best_params_', {})}")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Metrics")
        st.write(f"**Accuracy**: {acc:.4f}")
        st.text(classification_report(y_test, y_pred, target_names=["negative","positive"]))
        confusion_matrix_plot(y_test, y_pred, st)

        st.session_state["pending_model"] = model
        st.success("Model is ready to be saved in memory. See the section below")

st.markdown("---")
st.subheader("Save model to registry (in memory)")

if "pending_model" not in st.session_state:
    st.info("Train a model first (or load one) to enable saving.")
else:
    default_model_name = f"{model_name}_{ds_choice}" if "model_name" in locals() else st.session_state.get("last_model_name", "pipeline")
    new_name = st.text_input("Model name", value=st.session_state.get("last_model_name", default_model_name), key="save_name")
    if st.button("Save to memory"):
        st.session_state.setdefault("models", {})
        st.session_state["models"][new_name] = st.session_state["pending_model"]
        st.session_state["last_model_name"] = new_name
        st.success(f"Saved model as: {new_name}")

    # Optionnel : permettre le download du dernier modèle prêt
    from utils import bytes_downloadable_model
    buf = bytes_downloadable_model(st.session_state["pending_model"])
    st.download_button("Download .pkl", buf, file_name=f"{new_name}.pkl")

st.markdown("---")
st.subheader("Load an existing pipeline (.pkl) into registry")
up = st.file_uploader("Upload a joblib/pkl pipeline", type=["pkl"])
if up is not None:
    import io, joblib
    loaded = joblib.load(io.BytesIO(up.getvalue()))
    default_loaded_name = up.name.replace(".pkl","")
    loaded_name = st.text_input("Registry name for uploaded model", value=default_loaded_name, key="loaded_name")
    if st.button("Add uploaded model to registry"):
        st.session_state["models"][loaded_name] = loaded
        st.session_state["last_model_name"] = loaded_name
        st.success(f"Added model '{loaded_name}' to registry ✅")