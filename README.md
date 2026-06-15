# Analyse de Sentiments IMDB - Exploration des Technologies NLP

## **Présentation du projet**

Dans le cadre de mon Master 1 Ingénierie de l'Intelligence Artificielle à l’Université Paris 8, j'ai réalisé ce projet TER en autonomie afin d'explorer les technologies de Traitement du Langage Naturel (NLP). L’objectif est de concevoir un modèle capable de déterminer automatiquement si une critique de film est positive ou négative.

Le projet s'appuie sur la célèbre base de données IMDB (disponible sur Kaggle), contenant 50 000 critiques de films équilibrées. L'enjeu est de comparer l'impact de différentes méthodes de prétraitement, d'embedding et d'algorithmes (Machine Learning et Deep Learning) pour trouver le pipeline le plus performant.

## **Objectifs**
1. **Prétraiter** les données textuelles (SpaCy, NLTK) : suppression des stopwords, ponctuation, lemmatisation.
2. **Vectoriser** les textes (Embedding) : comparaison entre des approches basiques (TF-IDF) et contextuelles (Word2Vec).
3. **Entraîner et comparer** plusieurs modèles de classification :
   - *Machine Learning classique :* Régression Logistique, SVM, Naïve Bayes, Passive-Aggressive, Random Forest.
   - *Deep Learning :* LSTM, CNN.
4. **Optimiser** les hyperparamètres à l'aide de GridSearch.

## **Pipeline du projet**
Le flux de traitement suit une architecture claire :
- **Nettoyage :** Normalisation avec SpaCy et NLTK.
- **Représentation :** Génération de matrices TF-IDF ou vecteurs denses via Gensim (Word2Vec).
- **Classification :** Évaluation des différents modèles selon leur précision, rappel et F1-score.

## **Structure du projet**

L'architecture du projet s'organise autour de scripts utilitaires et de notebooks spécifiques à chaque modèle :

- `app.py` : script principal / application.
- `DeepLearning.py`, `Exploration.py`, `Test.py`, `Training.py` : scripts de conception et d'entraînement globaux.
- `dataVisualisation.py` : outils de visualisation des données.
- `metrics.py` : calcul des performances et matrices de confusion.
- `utils.py` : fonctions utilitaires partagées.
- `pretraitement.ipynb` : notebook dédié au nettoyage des données textuelles.
- `modelEvaluation.ipynb` : notebook d'évaluation globale des modèles.
- `main.ipynb` : notebook principal d'exploration.
- `training*.ipynb` : ensemble de notebooks dédiés à l'entraînement individuel et l'optimisation de chaque modèle :
  - `trainingCNN.ipynb` / `trainingCNNup.ipynb` (Réseaux de neurones convolutifs)
  - `trainingLSTM.ipynb` (Réseaux de neurones récurrents)
  - `trainingSVC.ipynb` (Support Vector Machine)
  - `trainingRegression.ipynb` / `trainingRegressionPipe.ipynb` (Régression Logistique)
  - `trainingNBM.ipynb` (Naïve Bayes)
  - `trainingPAC.ipynb` (Passive-Aggressive)
  - `trainingRF.ipynb` (Random Forest)
  - `trainingW2V.ipynb` (Entraînement de l'embedding Word2Vec)
- `requirements.txt` : liste des dépendances Python.
- `README.md` : présentation du projet.

## **Prérequis**

- Python 3.x (Attention à la compatibilité des versions pour Gensim et SpaCy).
- Un environnement virtuel est fortement recommandé (`venv` ou `conda`).
- Installer les dépendances listées dans `requirements.txt`.

## **Instructions pour reproduire le projet**

### Cloner le dépôt et préparer l'environnement
```bash
# Installation des dépendances
pip install -r requirements.txt
```
### Télécharger les modèles linguistiques SpaCy
```bash
python -m spacy download en_core_web_sm
```
### Exécution
Vous pouvez explorer pas à pas la démarche en ouvrant le fichier pretraitement.ipynb puis les différents training[Modele].ipynb via Jupyter Notebook ou Google Colab, ou bien lancer les scripts Python directement.

## Résultats clés
- Le CNN avec Word2Vec s'impose comme le meilleur modèle avec une accuracy d'environ 94%, captant parfaitement les motifs locaux ("not good", "very bad").
- Les modèles SVM et Régression Logistique avec TF-IDF offrent une excellente baseline très rapide, atteignant environ 90% de précision.
- Le modèle LSTM atteint 90,6%, offrant une bonne compréhension contextuelle bien que plus coûteux en temps d'entraînement.

## Remerciements
Merci d’avoir pris le temps de lire cette documentation. Ce projet a été extrêmement formateur sur le cycle complet de développement d'une IA en NLP. Bonne exploration du code source !

Djibril DAHOUB
