import os

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
# Time pour la mesure de temps d'exécution
import time

# Mesurer le temps de début
start_time = time.time()

# Chemins des données
train_dir = './data/train'
test_dir = './data/test'

# Paramètres de base
IMG_SIZE = (299, 299)
BATCH_SIZE = 8
EPOCHS = 20

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True
)

# Normalisation des images
# Création des générateurs d'images pour la validation
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Récupération des images d'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Récupération des images de test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Fonction permettant le calcul du F1-Score
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Arrondir les prédictions pour obtenir 0 ou 1
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  # Vrai positif
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)  # Vrai négatif
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  # Faux positif
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  # Faux négatif

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

# Construction du modèle CNN
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), 
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])

# Entraînement du modèle
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
#    callbacks=early_stopping,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator),
#    class_weight=class_weights_dict
)

# Récupération des données d'évaluation
loss, accuracy, f1 = model.evaluate(test_generator)

# Affichage des résultats finaux
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation F1-Score: {f1}")

# Création d'une liste de paramètre pour le nommage des fichiers
param = "covid_only_full_wh05_zs05_20E_8B"

# Sauvegarde du modèle
model.save(f"./results/{param}-{f1}_model.keras")

# Mesurer le temps de fin
end_time = time.time()

# Calculer et afficher la durée
execution_time = end_time - start_time
print(f"Le modèle a pris {execution_time:.4f} secondes pour s'entraîner, ")

# Convertir le temps en heures, minutes et secondes
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = int(execution_time % 60)

# Afficher le résultat
print(f"soit {hours} heures, {minutes} minutes et {seconds} secondes.")

# Tracer les courbes de précision
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Enregistrer l'image
plt.savefig(f'./results/perf_{param}_accuracy.png')
plt.show()

# Tracer les courbes de perte
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Enregistrer l'image
plt.savefig(f'./results/perf_{param}_loss.png')
plt.legend()
plt.show()

# Tracer les courbes de perte
plt.plot(history.history['f1_score'], label='train f1-score')
plt.plot(history.history['val_f1_score'], label='val f1-score')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
# Enregistrer l'image
plt.savefig(f'./results/perf_{param}_f1.png')
plt.legend()
plt.show()

# Créer ou ouvrir le fichier en mode ajout
with open(f"./results/log_{param}-{f1}.txt", "a") as fichier:
    fichier.write(f"Durée de l'entraînement : {hours} heures, {minutes} minutes et {seconds} secondes.\n")
    fichier.write(f"Validation Loss : {loss}.\n")
    fichier.write(f"Validation Accuracy : {accuracy}.\n")
    fichier.write(f"Validation F1-Score : {f1}.\n")
