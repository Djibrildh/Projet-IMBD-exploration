import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils import parse_text_columns, plot_class_balance

st.title("Exploration")

st.session_state.setdefault("datasets", {})   # name -> {"df": df, "label_col": label}
st.session_state.setdefault("last_dataset_name", None)

uploaded = st.file_uploader("📥 Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded:
    for up in uploaded:
        df = pd.read_csv(up)
        df, text_col, label_col = parse_text_columns(df)
        if "__text__" not in df.columns:
            st.error(f"[{up.name}] Could not detect a text column. Expecting 'clean_review' or 'review'. Skipped.")
            continue

        default_name = up.name
        with st.form(f"form_{up.name}"):
            ds_name = st.text_input("Dataset name", value=default_name, key=f"name_{up.name}")
            submitted = st.form_submit_button("Save dataset")
        if submitted:
            if not label_col:
                label_col = st.selectbox("Pick label column", options=df.columns, key=f"label_{up.name}")
            # Save to session
            st.session_state["datasets"][ds_name] = {"df": df, "label_col": label_col}
            st.session_state["last_dataset_name"] = ds_name
            st.success(f"Saved dataset as: {ds_name}")

st.markdown("---")

if st.session_state["datasets"]:
    st.subheader("Available datasets")
    names = list(st.session_state["datasets"].keys())
    pick = st.selectbox("Preview dataset", names, index=names.index(st.session_state["last_dataset_name"]) if st.session_state["last_dataset_name"] in names else 0)
    ds = st.session_state["datasets"][pick]
    df = ds["df"]; label_col = ds["label_col"]

    st.write("Preview:")
    cols = ["__text__", label_col] if label_col in df.columns else ["__text__"]
    st.dataframe(df[cols].head(10))

    st.subheader("Quick stats")
    st.write(f"- Rows: **{len(df)}**")
    if label_col:
        plot_class_balance(df[label_col], st)

    st.subheader("Text length distribution")
    lengths = df["__text__"].astype(str).apply(lambda s: len(s.split()))
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=50)
    ax.set_title("Length (words)"); ax.set_xlabel("words"); ax.set_ylabel("frequency")
    st.pyplot(fig)
else:
    st.info("Upload one or more CSVs. Each saved dataset will appear here and be selectable on the Training page.")