import pandas as pd
import numpy as np

from sklearn.decomposition import IncrementalPCA
import cv2
import pickle

from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

from .get_absolute_path import get_absolute_path
from pathlib import Path
import os

def flattened_images(metadatas : pd.DataFrame, img_shape = (299, 299)) -> pd.DataFrame:
    """
    Cette fonction retourne un DataFrame pandas avec autant de lignes que d'images. Chaque ligne représente une image applati.
    Inputs:
        metadatas : Doit avoir autant de lignes que d'image, et avoir une colonne 'IMG_MASKED_URL' qui représente l'URL relatif des images masquées à partir de la racine du projet
        img_shape : (Défaut : (299, 299)) tuple qui contient les dimensions des images
    Output:
        df : DataFrame qui contient les images applaties (une image = une ligne)
    """
    # Initialisation
    columns = [f"pixel_{i}" for i in range(img_shape[0] * img_shape[1])]
    data = np.zeros((len(metadatas), img_shape[0] * img_shape[1])) # temporaire

    for line, img_url in enumerate(metadatas['IMG_MASKED_URL']):
        img_url = get_absolute_path(img_url)
        img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)

        flattened = img.ravel() # Retourne l'image applati

        data[line] = flattened
        print("img", line)

    df_flat = pd.DataFrame(data, columns=columns).astype(np.uint8)
    return df_flat

def get_stats(df_imgs : pd.DataFrame) -> pd.DataFrame:
    """
    Cette fonction retourn un dataFrame qui contient certaines statistiques liées aux images en entrée.
    Input:
        df_imgs : DataFrame contenant les images applaties (une ligne = une image)
    Output:
        df : DataFrame contenant les statistiques des images en entrées, chaque ligne représentant une image en entrée (ordre conservé)    
    """
    data = np.zeros((len(df_imgs), 8))
    df = pd.DataFrame(data, columns=['mean_density', 'std', 'skewness', 'kurtosis', 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_entropy'])
    for index, img in df_imgs.iterrows():
        img_lung = img[img != 0] # Isolement de la partie des poumons

        # Calcul de la densité moyenne
        mean_density = np.mean(img_lung)
        df.loc[index, 'mean_density'] = mean_density

        # Normalisation et applatissement de l'image
        img_normalized = img_lung / mean_density # Normalisation de l'image par rapport à sa densité moyenne

        # Calcul de statistiques liées à l'image
        df.loc[index, 'std'] = np.std(img_lung) # écart-type
        df.loc[index, 'skewness'] = skew(img_normalized) # Calcul de la skewness (mesure d'une asymétrie de la distribution qui peut indiquer des zones sombres ou claires)
        df.loc[index, 'kurtosis'] = kurtosis(img_normalized, fisher=True) # Calcul du kurtosis (indication sur la largeur des queues de la distribution qui peut renseigner sur valeurs extrêmes ou non)

        # Conversion de l'image flat en 2D pour la partie GLCM
        nb_px = np.sqrt(len(img)).astype('int')
        img_2D = img.to_numpy().reshape([nb_px, nb_px])

        # Calcul de la matrice GLCM et des caractéristiques associéeds
        glcm = graycomatrix(img_2D, distances=[1], angles=[0], levels=256)
        df.loc[index, 'glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0] # Indication sur le niveau de variation des contraste
        df.loc[index, 'glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0] # Indication sur l'homogénéité des textures
        df.loc[index, 'glcm_energy'] = graycoprops(glcm, 'energy')[0, 0] # Indication sur la régularité des textures
        df.loc[index, 'glcm_entropy'] = graycoprops(glcm, 'entropy')[0, 0] # Indication sur le niveau de désordre dans l'image

    return df

def fit_and_save_scaler(scaler, df : pd.DataFrame, path : str, file_name : str) -> None:
    """
    Cette fonction entraîne et enregistre un scaler donné en entrée (par exemple StandardScaler de sklearn).
    Inputs:
        scaler : Instance du scaler à entraîner
        df : DataFrame sur lequel le scaler doit s'entraîner
        path : Chemin relatif (à partir de la racine du projet) où enregistrer le scaler
        file_name : Nom du fichier d'enregistrement (doit être en .pkl)
    """
    scaler.fit(df) # Entraînement du scaler

    # Création du chemin d'enregistrement du scaler
    full_path = Path(path, file_name)
    full_path = get_absolute_path(full_path)

    # Enregistrement du scaler
    with open(full_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler enregistré à : {full_path}")
    return

def apply_scaler(df : pd.DataFrame, path) -> pd.DataFrame:
    """
    Cette fonction applique un scaler (de type StandardScaler de sklearn) à df.
    Inputs:
        df : DataFrame sur lequel le scaler doit être appliqué
        path : Chemin relatif (à partir de la racine du projet) où le scaler est enregistré. Le chemin doit contenir le nom du fichier (.pkl)
    Outputs:
        df : DataFrame d'entrée après application du scaler
    """
    full_path = get_absolute_path(path)

    # Ouverture du scaler
    with open(full_path, 'rb') as f:
        scaler = pickle.load(f)

    # Application du scaler aux données
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    print(f"Scaler appliqué aux données : {full_path}")
    return df_scaled

def fit_and_save_encoder(encoder, y : pd.Series, path : str, file_name : str) -> None:
    """
    Cette fonction entraîne et enregistre un encoder donné en entrée (par exemple LabelEncoder de sklearn).
    Inputs:
        encoder : Instance de l'encodeur à entraîner
        y : Série sur laquelle l'encodeur doit s'entraîner
        path : Chemin relatif (à partir de la racine du projet) où enregistrer l'encodeur
        file_name : Nom du fichier d'enregistrement (doit être en .pkl)
    """
    encoder.fit(y) # Entraînement du scaler

    # Création du chemin d'enregistrement du scaler
    full_path = Path(path, file_name)
    full_path = get_absolute_path(full_path)

    # Enregistrement du scaler
    with open(full_path, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"Encoder enregistré à : {full_path}")
    return

def apply_encoder(y : pd.Series, path) -> pd.Series:
    """
    Cette fonction applique un encodeur (de type LabelEncoder de sklearn) à y.
    Inputs:
        y : Série sur laquelle l'encodeur doit être appliqué
        path : Chemin relatif (à partir de la racine du projet) où l'encodeur est enregistré. Le chemin doit contenir le nom du fichier (.pkl)
    Outputs:
        y_encoded : Série d'entrée après application de l'encodeur
        encoder : instance de l'encodeur
    """
    full_path = get_absolute_path(path)

    # Ouverture du scaler
    with open(full_path, 'rb') as f:
        encoder = pickle.load(f)

    # Application du scaler aux données
    y_scaled = encoder.transform(y)

    print(f"Encodeur appliqué aux données : {full_path}")
    return (y_scaled, encoder)

def fit_and_save_ipca(ipca, batch_size, df : pd.DataFrame, path, file_name : str) -> pd.DataFrame:
    """
    Cette fonction entraîne et enregistre un IncrementalPCA.
    Inputs:
        ipca : Instance de l'IPCA à entraîner
        batch_size : Taille des paquets à utiliser lors de l'apprentissage
        df : DataFrame sur lequel l'IPCA doit s'entraîner
        path : Chemin relatif (à partir de la racine du projet) où enregistrer l'IPCA
        file_name : Nom du fichier d'enregistrement (doit être en .pkl)
    """
    print(f"Données sur lesquelles le PCA a été entraîné : 0")
    for i in range(0, len(df), batch_size):
        end_index = min(i+batch_size, len(df)) # Permet de ne pas dépasser l'index de fin
        batch = df.iloc[i:end_index, :]
        ipca.partial_fit(batch)
        print(f"Données sur lesquelles le PCA a été entraîné : {end_index}")

    # Création du chemin d'enregistrement du scaler
    full_path = Path(path, file_name)
    full_path = get_absolute_path(full_path)

     # Enregistrement du pca
    with open(full_path, 'wb') as f:
        pickle.dump(ipca, f)

    print(f"PCA enregistré à : {full_path}")
    return

def apply_pca(df : pd.DataFrame, path) -> (pd.DataFrame, IncrementalPCA):
    """
    Cette fonction applique un PCA à df.
    Inputs:
        df : DataFrame sur lequel appliquer le PCA
        path : Chemin relatif (à partir de la racine du projet) où le PCA est enregistré. Le chemin doit contenir le nom du fichier (.pkl)
    Outputs:
        df : DataFrame d'entrée après application du PCA
        pca : instance du PCA, si besoin d'accès aux attributs de celle-ci
    """
    full_path = get_absolute_path(path)

    # Ouverture du PCA
    with open(full_path, 'rb') as f:
        pca = pickle.load(f)

    # Application du PCA aux données
    df_pca = pca.transform(df)
    columns = [f"PCA_{i+1}" for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(df_pca, columns=columns)

    print(f"PCA appliqué aux données : {full_path}")
    return (df_pca, pca)
