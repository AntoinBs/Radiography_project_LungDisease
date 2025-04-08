import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
import cv2

import tkinter as tk
from tkinter import filedialog

from src.features.get_absolute_path import get_absolute_path


def preprocessing(image_path, target_size=(299,299)):
    """
    Fonction qui réalise les étapes de preprocessing nécessaire au modèle que nous avons créé.
    Les étapes sont les suivantes :
        Ouverture de l'image
        Resize de l'image aux dimension 299 x 299 px
        Ajout d'une dimension à l'image qui permet de simuler un batch en entrée du modèle
    NB : La normalisation de l'image entre [-1, 1] est réalisé directement dans les couches du modèles, il n'est donc pas nécessaire
    de le faire ici.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Redimensionner
    return np.expand_dims(image, axis=0)  # Ajouter batch dimension

def predict_images(model, path, class_names, labeled_images = False):
    
    path = get_absolute_path(path)

    predictions = []
    if labeled_images:
        for label in path.glob("*"):
            path = Path(path, label)
            for img_file in path.glob("*.png"):
                print(img_file.name)
                image = preprocessing(img_file)
                prediction = np.argmax(model.predict(image), axis=1)
                prediction = class_names[prediction[0]]
                predictions.append([img_file, label.name, prediction])
        predictions = pd.DataFrame(predictions, columns=['FILE', 'TRUE_LABEL', 'PREDICTED_LABEL'])
    else:
        for img_file in path.glob("*.png"):
                print(img_file.name)
                image = preprocessing(img_file)
                prediction = np.argmax(model.predict(image), axis=1)
                prediction = class_names[prediction[0]]
                predictions.append([img_file, prediction])
        predictions = pd.DataFrame(predictions, columns=['FILE', 'PREDICTED_LABEL'])
    return predictions

def predict_one_image(model, class_names, file_path = None):
    if file_path == None:
        root = tk.Tk()
        root.withdraw()  # Cacher la fenêtre principale
        file_path = filedialog.askopenfilename(title="Sélectionner une image")
    file_path = get_absolute_path(file_path)
    image = preprocessing(file_path)
    prediction = np.argmax(model.predict(image), axis=1)
    prediction = class_names[prediction[0]]
    prediction = pd.DataFrame([[file_path, prediction]], columns=['FILE', 'PREDICTED_LABEL'])
    return prediction

def show_predictions(predictions, one_image : bool = False):

    if not one_image:
        lignes_subplot = (len(predictions) // 5) + 1
        plt.figure(figsize=(25, lignes_subplot*5))
        for i, (file, prediction) in enumerate(zip(predictions['FILE'], predictions['PREDICTED_LABEL'])):
            plt.subplot(lignes_subplot,5,i+1)
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            plt.imshow(image, cmap='gray')
            plt.title(f"Prédiction : {prediction}", fontsize=14, fontweight="bold")
            plt.axis('off')
        plt.show()
    else:
        plt.figure(figsize=(5, 5))
        image = cv2.imread(predictions['FILE'][0], cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap='gray')
        plt.title(f"Prédiction : {predictions['PREDICTED_LABEL'][0]}", fontsize=14, fontweight="bold")
        plt.axis('off')
        plt.show()