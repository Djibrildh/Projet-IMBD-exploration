from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

RANDOM_STATE = 42

def testMLModel(model, X_train, y_train, X_test, y_test, title=""):
    # Prédictions sur le test
    y_pred = model.predict(X_test)

    # Rapport de classification
    print(f"\n=== {title} : Rapport de classification ===")
    print(classification_report(y_test, y_pred, digits=4))

    # Matrice de confusion
    print(f"=== {title} : Matrice de confusion ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Négatif","Positif"],
                yticklabels=["Négatif","Positif"])
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(f"Matrice de confusion - {title}")
    plt.show()

    # Courbe d'apprentissage (Accuracy)
    print(f"=== {title} : Courbe d'apprentissage (accuracy) ===")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train accuracy")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation accuracy")
    plt.xlabel("Taille d'entraînement"); plt.ylabel("Accuracy"); plt.title(f"Learning curve - {title}")
    plt.legend(); plt.show()


def testDLModels(model, X_pad, y_true, history=None, class_names=None, title="Évaluation"):
    """
    X_pad : tableau numpy des séquences PAD (résultat de to_pad)
    y_true: labels vérité terrain -> entiers (0/1/2/..), ou one-hot de forme (n, k)
    history: objet Keras History OU None (si tu n'as pas l'entraînement sous la main)
    class_names: liste ["neg","pos"] ou ["classe 0", "classe 1", ...]
    """

    # ---- 3.1 Normaliser y_true en entiers ----
    y_true = np.asarray(y_true)
    if y_true.ndim == 2 and y_true.shape[1] > 1:   # one-hot -> entiers
        y_true = y_true.argmax(axis=1)
    else:
        y_true = y_true.ravel().astype(int)

    # ---- 3.2 Prédire ----
    y_pred_raw = model.predict(X_pad, verbose=0)

    # ---- 3.3 Convertir prédictions en classes ----
    if y_pred_raw.ndim == 1 or y_pred_raw.shape[1] == 1:
        # binaire (sigmoïde)
        y_pred = (y_pred_raw.ravel() >= 0.5).astype(int)
        if class_names is None: class_names = ["negative","positive"]
    else:
        # multi-classe (softmax)
        y_pred = y_pred_raw.argmax(axis=1)
        if class_names is None:
            class_names = [f"classe {i}" for i in range(y_pred_raw.shape[1])]

    # ---- 3.4 Métriques ----
    is_binary = (len(np.unique(y_true)) == 2)
    avg = "binary" if is_binary else "weighted"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)

    print(f"\n=== {title} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision ({avg}) : {prec:.4f}")
    print(f"Recall    ({avg}) : {rec:.4f}")
    print(f"F1-score  ({avg}) : {f1:.4f}\n")

    print("Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ---- 3.5 Matrice de confusion ----
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion - {title}")
    plt.xlabel("Prédit"); plt.ylabel("Vrai"); plt.show()

    # ---- 3.6 Courbes entraînement (si history fourni) ----
    if history is not None and hasattr(history, "history"):
        h = history.history
        plt.figure(figsize=(10,4))
        # perte
        plt.subplot(1,2,1)
        if "loss" in h: plt.plot(h["loss"], label="Train")
        if "val_loss" in h: plt.plot(h["val_loss"], label="Val")
        plt.title("Perte"); plt.xlabel("Epochs"); plt.legend()
        # accuracy
        plt.subplot(1,2,2)
        if "accuracy" in h: plt.plot(h["accuracy"], label="Train")
        if "val_accuracy" in h: plt.plot(h["val_accuracy"], label="Val")
        plt.title("Accuracy"); plt.xlabel("Epochs"); plt.legend()
        plt.suptitle(title)
        plt.tight_layout(); plt.show()

    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "cm":cm}

def pretraitement_review(model, text, tokenizer, class_names, max_len=200):
    """
    text : str, une critique (ex: "This movie was awful !")
    return : label ("positive"/"negative") et la proba associée
    """
    # Toujours travailler sur une liste, même pour un seul texte
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)

    # Prédiction
    proba = model.predict(pad, verbose=0)

    # Cas binaire (sigmoïde)
    if proba.shape[1] == 1:
        proba_positive = proba[0][0]
        label = class_names[1] if proba_positive >= 0.5 else class_names[0]
        proba_label = proba_positive if label == class_names[1] else 1 - proba_positive

    # Cas multi-classes (softmax)
    else:
        label_idx = np.argmax(proba[0])
        label = class_names[label_idx]
        proba_label = proba[0][label_idx]

    return label, float(proba_label)

def to_pad(texts, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len)