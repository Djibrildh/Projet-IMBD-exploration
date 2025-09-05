import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def viewData(csvFilePath):
    """Show information about the data:
        - List of columns
        - Number of lines
        - First 10 lines in the data file
    """
    df = pd.read_csv(csvFilePath)
    print("collonnes : ",df.columns)
    print("nombre de ligne : ", df.shape[0])
    print("premières lignes du fichier : ")
    print(df.head())

def viewDistrib(csvFilePath):
    """
        Show The distribution
    """
    df = pd.read_csv(csvFilePath)
    labels = df["sentiment"]
    labelsCount = Counter(labels)

    print("Répartition des labels :")
    for label, count in labelsCount.items():
        print(f"{label}: {count} images")

    plt.figure(figsize=(10, 5))
    plt.bar(labelsCount.keys(), labelsCount.values(), color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.title("Répartition des labels dans le dataset")
    plt.xticks(rotation=45)
    plt.show()