import pandas as pd
import numpy as np

from src.features.get_absolute_path import get_absolute_path

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, RandomRotation, RandomFlip, RandomZoom, RandomContrast, Rescaling
from tensorflow.keras.models import Model

from tensorflow.keras.applications import Xception

def open_datas(train_folder, validation_folder, test_folder, batch_size=32, image_size=(299,299)):
    """
    Cette fonction ouvre les données d'entraînement, de validation et de test sous forme de datasets. Les données doivent préalablement 
    avoir été enregistrées dans des dossiers spécifiques.
    Outputs:
        train_ds : dataset d'entraînement
        val_ds : dataset de validation
        test_ds : dataset de test
        class_names : liste des noms des classes associés à l'encodage de la target (index de la liste = encodage créé sur la target)
    """
    train_ds = image_dataset_from_directory(train_folder,
                                            label_mode="int",
                                            shuffle=True,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            seed=42)

    val_ds = image_dataset_from_directory(validation_folder,
                                        label_mode="int",
                                        shuffle=True,
                                        batch_size=batch_size,
                                        image_size=image_size,
                                        seed=42)

    test_ds = image_dataset_from_directory(test_folder,
                                        label_mode="int",
                                        shuffle=True,
                                        batch_size=batch_size,
                                        image_size=image_size,
                                        seed=42)
    class_names = train_ds.class_names

    # Optimisation du chargement
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def calcul_weight_dict(ds, class_names):
    """
    Cette fonction calcul les poids ajustés à la répartition des classes, pour combler le déséquilibrage de celles-ci durant 
    l'entraînement
    Outputs:
        class_weight_dict : dictionnaire des poids 
    """
    labels = np.concatenate([y for _, y in ds], axis=0)

    # Calcul des poids des classes
    class_labels = np.unique(labels)
    class_weight = compute_class_weight(class_weight='balanced', classes=class_labels, y=labels)
    class_weight_dict = dict(enumerate(class_weight))

    print(f"Poids des classes :")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {class_weight_dict[i]}")
              
    return class_weight_dict

def train_model(train_ds, val_ds, class_weights, batch_size=32, epochs=100, save_model_path=r"models\model_xception.keras"):
    """
    Fonction qui lance l'entraînement du modèle et l'enregistre à l'emplacement save_model_path
    Output:
        model : instance du modèle entraîné
        combined_history : historique d'entraînement avant et après libération des couches de Xception
    """
    save_model_path = get_absolute_path(save_model_path)

    # Définition des callbacks
    modelcheckpoint = ModelCheckpoint(filepath=save_model_path,
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    mode="min")

    earlystop = EarlyStopping(monitor="val_loss",
                            mode="min",
                            min_delta=0,
                            patience=6,
                            verbose=1,
                            restore_best_weights=True)

    reducelr = ReduceLROnPlateau(monitor="val_loss",
                                min_delta=0.001,
                                patience=3,
                                factor=0.5,
                                cooldown=2,
                                verbose=1)
    
    # Modèle Xception
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3), pooling="max")

    # Freeze des couches du Xception
    base_model.trainable = False

    # Couche d'entrée
    inputs = Input(shape=(299, 299, 3))

    # Data augmentation
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1, fill_mode='constant', fill_value=0)(x)
    x = RandomZoom(0.2, fill_mode='constant', fill_value=0)(x)
    x = RandomContrast(0.2)(x)

    # Scaling
    x = Rescaling(scale=1./127.5, offset=-1)(x)

    # Construction du modèle
    x = base_model(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history_model = model.fit(train_ds,
                              validation_data=val_ds,
                              batch_size=batch_size,
                              epochs=epochs,
                              class_weight=class_weights,
                              callbacks=[earlystop, reducelr, modelcheckpoint])
    stopped_epoch = np.argmin(history_model.history['val_loss'])

    
    print('-----------------------------------------------------------')
    print('------------------------FINE-TUNING------------------------')
    print('-----------------------------------------------------------')
    
    for layer in base_model.layers[:]:
        layer.trainable = True

    history_model_2 = model.fit(train_ds,
                              validation_data=val_ds,
                              batch_size=batch_size,
                              epochs=epochs,
                              initial_epoch=stopped_epoch,
                              class_weight=class_weights,
                              callbacks=[earlystop, reducelr, modelcheckpoint])

    combined_history = {key: history_model.history[key][:stopped_epoch] + history_model_2.history[key] for key in history_model.history.keys()}

    return model, combined_history
