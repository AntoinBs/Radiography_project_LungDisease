import pandas as pd

import os
from pathlib import Path
import shutil

from src.features.add_url import add_url
from src.features.get_absolute_path import get_absolute_path

import cv2

from sklearn.model_selection import train_test_split

def create_metadata():
    """
    Cette fonction créer un seul dataframe de métadonnées à partir des quatres fichiers excels du projet.
    Elle enregistre le dataframe résultant sous forme de fichier csv au chemin suivant : data\processed\metadatas.csv
    Output:
        df : fichier de metadonnées
    """
    df_covid = pd.read_excel(get_absolute_path(r"data\raw\COVID.metadata.xlsx"))
    df_normal = pd.read_excel(get_absolute_path(r"data\raw\Normal.metadata.xlsx"))
    df_viral_pneumonia = pd.read_excel(get_absolute_path(r"data\raw\Viral Pneumonia.metadata.xlsx"))
    df_lung_opacity = pd.read_excel(get_absolute_path(r"data\raw\Lung_Opacity.metadata.xlsx"))

    # Ajout d'une colonne LABEL pour identifier le type de pathologie
    for df, name in zip([df_covid, df_normal, df_viral_pneumonia, df_lung_opacity], ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA', 'LUNG_OPACITY']):
        df['LABEL'] = name

    # Rassemblement des 4 dataframe en un seul
    df = pd.concat([df_covid, df_normal, df_viral_pneumonia, df_lung_opacity], axis=0, ignore_index=True)

    # Ajout des urls des images et masques
    df = add_url(df)
    df = add_url(df, file_type='mask')

    # Enregistrement du dataframe des métadonnées
    df.to_csv(get_absolute_path(r'data\processed\metadatas.csv'), sep=',', encoding='utf-8', index=False, header=True)
    return df

def save_img(img : cv2.imread, file_name : str, path : str):
        """ 
        Fonction qui enregistre une image à l'endroit spécifié, avec le nom file_name
        """
        path = get_absolute_path(path)
        try :
            initial_dir = Path(__file__).resolve().parent # __file__ is accessible in .py files but not in .ipynb files
        except NameError:
            initial_dir = Path(os.getcwd()).resolve() # for .ipynb files
        os.chdir(path)
        cv2.imwrite(file_name, img)
        initial_dir = get_absolute_path(initial_dir)
        os.chdir(initial_dir)

import os
import pandas as pd

def rename_files(metadatas: pd.DataFrame) -> pd.DataFrame:
    # Créer une copie pour éviter de modifier l'original
    updated_metadatas = metadatas.copy()

    for label in updated_metadatas['LABEL'].unique(): 
        # Gérer les images
        path_images = f'data/raw/{label}/images'
        filenames_images = os.listdir(path_images)
        for filename in filenames_images:
            new_filename = filename.replace(' ', '_')
            os.rename(os.path.join(path_images, filename), os.path.join(path_images, new_filename))
            print(filename, 'renamed to', new_filename)
            # Mettre à jour le nom du fichier dans la DataFrame
            updated_metadatas.loc[
                (updated_metadatas['LABEL'] == label) & 
                (updated_metadatas['FILE NAME'] == filename), 
                'FILE NAME'
            ] = new_filename

        # Gérer les masques
        path_masks = f'data/raw/{label}/masks'
        filenames_masks = os.listdir(path_masks)
        for filename in filenames_masks:
            new_filename = filename.replace(' ', '_')
            os.rename(os.path.join(path_masks, filename), os.path.join(path_masks, new_filename))
            print(filename, 'renamed to', new_filename)
            # Mettre à jour le nom du fichier dans la DataFrame
            updated_metadatas.loc[
                (updated_metadatas['LABEL'] == label) & 
                (updated_metadatas['FILE NAME'] == filename), 
                'FILE NAME'
            ] = new_filename

    updated_metadatas['FILE NAME']=updated_metadatas['FILE NAME'].replace(' ', '_', regex=True)
    updated_metadatas.to_csv(r"data\processed\metadatas.csv")
    return updated_metadatas







def mask_images(metadatas : pd.DataFrame):
    """
    Fonction qui applique les masques aux images correspondantes à partir du fichier de métadonnées où sont stockés les URLs de celles-ci.
    Enregistre les images masquées à l'emplacement suivant : data/processed (à partir de la racine du projet)
    Enregistre également les URLs des images masquées dans le fichier des métadonnées et l'enregistre à l'emplacement suivant:
    data\processed\metadatas_with_url.csv
    Output:
        metadatas : dataframe des métadonnées avec les URLs des images masquées en plus
    """

    

        
        # Parcours de tous les masques, augmentation en 299x299, application aux image puis enregistrement des nouveaux masques augmentés
    for img_url, mask_url, file_name, label in zip(metadatas["IMG_URL"], metadatas["MASK_URL"], metadatas['FILE NAME'], metadatas['LABEL']):
        img_url = get_absolute_path(img_url) # get absolute path
        mask_url = get_absolute_path(mask_url) # get absolute path



        mask = cv2.imread(mask_url, cv2.IMREAD_GRAYSCALE)
        print(mask_url, 'read')
        img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
        print(img_url, 'read')
        # Augmentation du masque
        mask_resized = cv2.resize(mask, dsize=(299,299), interpolation=cv2.INTER_NEAREST)

        # Application du masque à l'image
        img_masked = cv2.bitwise_and(img, mask_resized)

        file_name = file_name + '.png'

        # Création du chemin absolu pour la sauvegarde de l'image masquées
        path = Path("data", "processed", label, "images_masked")
        path_abs = get_absolute_path(path) # convertion en chemin absolu
        # Créer le dossier de destination s'il n'existe pas
        path_abs.mkdir(parents=True, exist_ok=True)

        save_img(img_masked, file_name, path_abs)

    # Enregistrement des chemins d'accès dans les métadonnées
    for file, format, label in zip(metadatas['FILE NAME'], metadatas['FORMAT'], metadatas['LABEL']):
            format = format.lower() # set format in lowercase
            file_name = file + '.' + format # concatenate file and format to make global file_name
            path = Path("data", "processed", label, 'images_masked', file_name) # build Path from the variables in the current line of dataset
            metadatas.loc[(metadatas['FILE NAME'] == file), 'IMG_MASKED_URL'] = path # update the path in DataFrame

        
    # Enregistrement des métadonnées avec les liens des masques augmentés
    metadatas.to_csv(get_absolute_path(r'data\processed\metadatas_with_url.csv'), sep=',', encoding='utf-8', index=False, header=True)

    return metadatas

def create_train_test_folder(metadatas):
    """
    Cette fonction créer un dossier train_test_split à l'emplacement suivant: data\processed\train_test_split
    L'architecture au sein de ce dossier sera:
        |---test
        |     |---COVID
        |     |---LUNG OPACITY
        |     |---NORMAL
        |     |---VIRAL PNEUMONIA
        |---train
        |     |---COVID
        |     |--- ...
        |---validation
        |     |---COVID
        |     |--- ...
    Ainsi les images sont séparés à l'aide de la fonction train_test_split de sklearn.model_selection en gardant les proportions des classes
    """

    df = metadatas[['FILE NAME', 'LABEL', 'IMG_MASKED_URL']]

    # Train dataframe
    strat = df['LABEL']
    train_df, df = train_test_split(df, test_size=0.2, shuffle=True, stratify=strat, random_state=42)

    # Test et validation dataframe
    strat = df['LABEL']
    valid_df, test_df = train_test_split(df, test_size=0.5, shuffle=True, stratify=strat, random_state=42)

    for df, folder in zip([train_df, valid_df, test_df], ["train", "validation", "test"]):

        # Création du chemin de destination
        destination_dir = get_absolute_path(r"data\processed\train_test_split")
        destination_dir = Path(destination_dir, folder)
        
        for source_img_path, label in zip(df['IMG_MASKED_URL'], df['LABEL']):
            # Ajout du dossier correspondant à la classe dans le chemin de destination
            destination_dir_img = Path(destination_dir, label)
            
            # Créer le dossier de destination s'il n'existe pas
            destination_dir_img.mkdir(parents=True, exist_ok=True)

            # Création du chemin absolu à partir du chemin relatif de l'image source
            source_img_path = get_absolute_path(source_img_path)

            # Copie/colle de l'image source vers le dossier approprié
            shutil.copy(source_img_path, destination_dir_img/source_img_path.name)
            
            print(source_img_path.name)