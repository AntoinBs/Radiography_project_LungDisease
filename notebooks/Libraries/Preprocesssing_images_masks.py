import Libraries
from Libraries import add_url
from Libraries import get_absolute_path
import os
import cv2
import pandas as pd
from pathlib import Path
from Libraries import get_absolute_path
import shutil
import os
from pathlib import Path
import os
import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Fonction principale de prétraitement des images et masques
def preprocess_images(metadata: pd.DataFrame, mask_images: bool = True):
    
    def save_img(img, file_name, path):
        """Sauvegarde une image à un chemin spécifique."""
        initial_dir = Path(os.getcwd()).resolve()
        os.chdir(path)
        cv2.imwrite(file_name, img)
        os.chdir(initial_dir)

    # Permet d'ajouter les URLs des masks et images au tableau des métadonnées
    metadata['FILE NAME'] = metadata['FILE NAME'].str.replace('NORMAL', 'Normal')
    metadata['FILE NAME'] = metadata['FILE NAME'].str.replace(' ','_')

    metadata=add_url.add_url(df=metadata,file_type='img')
    metadata=add_url.add_url(df=metadata,file_type='mask')

    # Indicateur pour suivre les modifications
    modifications_effectuees = False

    # Résizer et sauvegarder les masques si mask_images est True
    for mask_url, file_name, label in zip(metadata["MASK_URL"], metadata['FILE NAME'], metadata['LABEL']):
            output_path = Path("../data/processed", label, "masks", file_name + ".png")

            if output_path.exists():
                print(f"Skipping {output_path}, already exists.")
                continue

            if not os.path.exists(mask_url):
                print(f"Warning: Mask file not found: {mask_url}")
                continue

            mask = cv2.imread(mask_url, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to load mask image: {mask_url}")
                continue

            mask_resized = cv2.resize(mask, dsize=(299, 299), interpolation=cv2.INTER_NEAREST)
            os.makedirs(output_path.parent, exist_ok=True)
            save_img(mask_resized, output_path.name, output_path.parent)
            print(f"Mask {output_path.name} saved")
            modifications_effectuees = True
            
            # Mise à jour des metadatas
            metadata.loc[metadata['FILE NAME'] == file_name, 'MASK_RESIZED_URL'] = str(output_path)

    if modifications_effectuees:
        metadata.to_csv(r'../data/processed/metadatas_with_url.csv', sep=',', encoding='utf-8', index=False, header=True)
    else:
        print("Aucune modification effectuée sur les masques.")
        metadata=pd.read_csv(r'../data/processed/metadatas_with_url.csv')

    
    # Masquer les images si mask_images est True
    if mask_images:
        for img_url, mask_url, img_name, label in zip(metadata['IMG_URL'], metadata['MASK_RESIZED_URL'], metadata['FILE NAME'], metadata['LABEL']):

            img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_url, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print(f"Erreur lors du chargement de l'image ou du masque: {img_url} ou {mask_url}")
                continue

            img_masked = cv2.bitwise_and(img, mask)
            file_name = img_name + ".png"
            path = Path("..", "data", "processed", label, "images_masked")
            path_with_file = Path(path, file_name)

            os.makedirs(path, exist_ok=True)
            save_img(img_masked, file_name, path)
            print(f"{img_name} sauvée")

            metadata.loc[metadata['FILE NAME'] == img_name, 'IMG_MASKED_URL'] = path_with_file

        metadata.to_csv(r'../data/processed/metadatas_with_url.csv')

    # Séparer les échantillons
    output_folder = "images_masked" if mask_images else "unmasked_images"
    df = metadata[['FILE NAME', 'LABEL', 'IMG_MASKED_URL' if mask_images else 'IMG_URL']]

    strat = df['LABEL']
    train_df, df = train_test_split(df, test_size=0.2, shuffle=True, stratify=strat, random_state=42)
    valid_df, test_df = train_test_split(df, test_size=0.5, shuffle=True, stratify=strat, random_state=42)

    for subset_df, folder in zip([train_df, valid_df, test_df], ['train', 'validation', 'test']):
        destination_dir = Path('../data/processed/train_test_split', folder, output_folder)
        os.makedirs(destination_dir, exist_ok=True)

        for source_img_path, label in zip(subset_df[output_folder.upper() + '_URL'], subset_df['LABEL']):
            destination_dir_img = Path(destination_dir, label)
            os.makedirs(destination_dir_img, exist_ok=True)

            shutil.copy(source_img_path, destination_dir_img/source_img_path.name)
            print(source_img_path.name)

    print("Traitement terminé.")
