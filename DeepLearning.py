import io
import re
import json
import joblib
import tempfile
import streamlit as st

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.title("🤖 Deep Learning — Load & Test (Keras)")

if not TF_AVAILABLE:
    st.warning("TensorFlow/Keras n'est pas installé. `pip install tensorflow keras`")
    st.stop()

def _minimal_clean(text: str):
    """
    ⚠️ Fallback minimal si tu n'importes pas ton vrai nettoyage.
    Remplace par TA fonction si possible pour l'iso parfaite.
    """
    txt = text.lower()
    txt = re.sub(r"[^a-z0-9\s']", " ", txt)
    tokens = txt.split()
    return tokens

def preprocess_like_training(raw_text: str):
    """
    Reproduit le flux du notebook :
      tokens -> ' '.join(tokens) -> texts_to_sequences
    """
    tokens = _minimal_clean(raw_text)
    return " ".join(tokens)

class KerasTextClassifier:
    """
    Petit wrapper pour uniformiser l'API avec la page Test :
      - predict(X) renvoie les labels
      - predict_proba(X) renvoie les probas (binaire ou softmax)
    """
    def __init__(self, keras_model, tokenizer, max_len=200, class_names=("negative","positive"),
                 padding="pre", truncating="pre"):
        self.model = keras_model
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.class_names = tuple(class_names)
        self.padding = padding
        self.truncating = truncating

    def _vectorize(self, texts):
        prepped = [preprocess_like_training(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(prepped)
        X = pad_sequences(
            seqs,
            maxlen=self.max_len,
            padding=self.padding,
            truncating=self.truncating
        )
        return X, prepped, seqs

    def predict_proba(self, texts):
        import numpy as np
        X, _, _ = self._vectorize(texts)
        proba = self.model.predict(X, verbose=0)
        if proba.ndim == 2 and proba.shape[1] == 1:
            p_pos = proba.ravel()
            p_neg = 1.0 - p_pos
            proba = np.vstack([p_neg, p_pos]).T
        return proba

    def predict(self, texts):
        import numpy as np
        proba = self.predict_proba(texts)
        idx = proba.argmax(axis=1)
        labels = [self.class_names[i] if i < len(self.class_names) else str(i) for i in idx]
        return labels

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    mdl_file = st.file_uploader("📥 Modèle Keras (.h5)", type=["h5"])
with col2:
    tok_file = st.file_uploader("📥 Tokenizer (.pkl)", type=["pkl"])
with col3:
    cfg_file = st.file_uploader("📄 (Optionnel) Config JSON", type=["json"])

cfg = {"max_len": 200, "class_names": ["negative", "positive"], "padding": "pre", "truncating": "pre"}

if cfg_file is not None:
    try:
        cfg.update(json.loads(cfg_file.getvalue().decode("utf-8")))
        st.caption("Config JSON chargée ✔️")
    except Exception as e:
        st.warning(f"Config JSON ignorée ({e})")

MODEL = None
TOKENIZER = None
INFO = {}

if mdl_file and tok_file:
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(mdl_file.getbuffer())
            mdl_path = tmp.name
        MODEL = load_model(mdl_path)

        TOKENIZER = joblib.load(io.BytesIO(tok_file.getvalue()))

        inferred_len = None
        try:
            shp = getattr(MODEL, "input_shape", None)
            if isinstance(shp, (list, tuple)) and len(shp) > 0:
                if isinstance(shp[0], (list, tuple)):
                    inferred_len = shp[0][1]
                else:
                    inferred_len = shp[1]
        except Exception:
            inferred_len = None

        max_len = int(cfg.get("max_len", 200))
        if inferred_len:
            max_len = int(inferred_len)

        class_names = tuple(cfg.get("class_names", ["negative", "positive"]))
        padding = cfg.get("padding", "pre")
        truncating = cfg.get("truncating", "pre")

        st.success("Modèle + tokenizer chargés ✅")

        with st.expander("Infos modèle"):
            st.write({
                "input_shape": getattr(MODEL, "input_shape", None),
                "max_len_used": max_len,
                "class_names": class_names,
                "padding": padding,
                "truncating": truncating
            })

        wrapped = KerasTextClassifier(
            keras_model=MODEL,
            tokenizer=TOKENIZER,
            max_len=max_len,
            class_names=class_names,
            padding=padding,
            truncating=truncating
        )

        st.subheader("🧪 Test rapide")
        default_text = "This movie is awesome, I loved it!"
        user_text = st.text_area("Entrez un texte :", value=default_text, height=100)

        colp1, colp2 = st.columns([1, 1])
        with colp1:
            if st.button("Prédire"):
                prepped = preprocess_like_training(user_text)
                seq = TOKENIZER.texts_to_sequences([prepped])[0]
                X = pad_sequences([seq], maxlen=wrapped.max_len, padding=wrapped.padding, truncating=wrapped.truncating)
                proba = MODEL.predict(X, verbose=0)

                if proba.ndim == 2 and proba.shape[1] == 1:
                    p_pos = float(proba.ravel()[0])
                    label = class_names[1] if p_pos >= 0.5 else class_names[0]
                    p_show = p_pos if label == class_names[1] else 1.0 - p_pos
                else:
                    import numpy as np
                    idx = int(proba[0].argmax())
                    label = class_names[idx] if idx < len(class_names) else str(idx)
                    p_show = float(np.max(proba[0]))

                st.success(f"Predicted sentiment: **{label}** (p={p_show:.3f})")

                with st.expander("🔎 Debug pretreatement"):
                    st.write({
                        "preprocessed_text": prepped,
                        "seq_len": len(seq),
                        "first_20_ids": seq[:20],
                        "max_len_used": wrapped.max_len
                    })

        with colp2:
            st.markdown("### Add register (for test page)")
            st.session_state.setdefault("models", {})
            default_name = (mdl_file.name or "keras_model").replace(".h5", "")
            reg_name = st.text_input("Model name in register", value=default_name, key="dl_reg_name")

            if st.button("Add to register"):
                st.session_state["models"][reg_name] = wrapped
                st.session_state["last_model_name"] = reg_name
                st.success(f"Model **{reg_name}** added to register")

    except Exception as e:
        st.error(f"Load/predict error: {e}")
else:
    st.info("add a **.h5** (modèle Keras) and a **.pkl** (tokenizer) file to start.")