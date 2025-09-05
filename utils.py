import io
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def parse_text_columns(df):
    """
    Try to infer a text column and a label column.
    - if 'clean_review' existe: we parse the list of tokens and join them
    - else 'review' existe: no problem
    - Label: search for 'sentiment'
    """
    text_col = None
    if "clean_review" in df.columns:
        try:
            texts = df["clean_review"].apply(lambda x: " ".join(ast.literal_eval(x)))
            text_col = "clean_review"
            df = df.copy()
            df["__text__"] = texts
        except Exception:
            pass

    if text_col is None:
        for c in df.columns:
            if c.lower() in ("review", "text", "comment"):
                text_col = c
                df = df.copy()
                df["__text__"] = df[c].astype(str)
                break

    label_col = None
    for c in df.columns:
        if c.lower() in ("sentiment", "label", "target", "y"):
            label_col = c
            break

    return df, text_col, label_col

def plot_class_balance(labels, title="Classes distribution"):
    vc = labels.value_counts().sort_index()
    fig, ax = plt.subplots()
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Example number")
    st.pyplot(fig)

def confusion_matrix_plot(y_true, y_pred, st, labels=("negative","positive"), title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)

def build_estimator(name, proba=False):
    """
    Returns a sklearn estimator according to the choice.
    proba=True for SVC with probability (slower).
    """
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    if name == "SVC":
        if proba:
            return SVC(kernel="linear", probability=True, random_state=42)
        return SVC(kernel="linear", probability=False, random_state=42)
    if name == "NaiveBayes (Multinomial)":
        return MultinomialNB()
    if name == "PassiveAggressive":
        return PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    raise ValueError("unknown model")

def default_param_grid(name, proba=False):
    """
    Grids for GridSearch/RandomizedSearch. (Voluntarily less ambitious)
    """
    if name == "LogisticRegression":
        return {"clf__C":[0.5,1.0,2.0], "clf__penalty":["l2"], "clf__solver":["liblinear","lbfgs"]}
    if name == "SVC":
        if proba:
            return {"clf__C":[0.5,1.0,2.0]}
        return {"clf__C":[0.5,1.0,2.0]}
    if name == "NaiveBayes (Multinomial)":
        return {"clf__alpha":[0.1,0.5,1.0]}
    if name == "PassiveAggressive":
        return {"clf__C":[0.5,1.0,2.0], "clf__loss":["hinge","squared_hinge"]}
    return {}

'''
A pipeline is a way to connect multiple steps of a Machine Learning workflow into one single object.
For example:
- Preprocessing (like TF-IDF)
- Model training (like SVM)
Instead of running each step separately, a pipeline makes them work together automatically.
'''
def make_pipeline(estimator, max_features=20000, use_bigrams=True):
    ngram = (1,2) if use_bigrams else (1,1)
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram)),
        ("clf", estimator)
    ])
    return pipe

def bytes_downloadable_model(pipeline):
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)
    return buf

class KerasTextClassifier:
    """
    Wraps a Keras model + tokenizer to expose sklearn-like API:
    - predict(texts) -> array of class ids {0,1}
    - predict_proba(texts) -> array of probabilities shape (n_samples, 2) for binary
    """
    def __init__(self, keras_model, tokenizer, max_len=200, class_names=("negative","positive")):
        self.model = keras_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.class_names = class_names

        # lazy import to avoid TF requirement on ML-only paths
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        self._pad_sequences = pad_sequences

    def _prepare(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        pad = self._pad_sequences(seq, maxlen=self.max_len)
        return pad

    def predict(self, texts):
        X = self._prepare(texts)
        proba = self.model.predict(X, verbose=0)
        # handle sigmoid (n,1) vs softmax (n,2)
        if proba.ndim == 2 and proba.shape[1] == 1:
            y = (proba.ravel() >= 0.5).astype(int)
        else:
            y = np.argmax(proba, axis=1).astype(int)
        return y

    def predict_proba(self, texts):
        X = self._prepare(texts)
        proba = self.model.predict(X, verbose=0)
        # convert sigmoid (n,1) to 2-col proba
        if proba.ndim == 2 and proba.shape[1] == 1:
            p1 = proba.ravel()
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        return proba