import streamlit as st
import pandas as pd
import numpy as np

# Librairie pour l'affichage des graphiques
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Librairie pour utiliser les dossiers du système
import os
from pathlib import Path

# Librairie pour générer un choix aléatoire
import random
from PIL import Image

# Librairies pour l'utilisation du modèle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, SeparableConv2D, RandomRotation, RandomFlip, RandomZoom, RandomContrast

# Librairie pour mesurer le temps écoulé
import asyncio

# Librairie pour la génération et la modification d'images
import cv2

# Fonction permettant d'afficher les points de chaleur des caractéristiques de l'image interprétés par le modèle
def grad_cam(img, model, base_model, layer_name : str):
    # Sélection de la couche dans le bon modèle
    layer = base_model.get_layer(layer_name)
    grad_model = Model(inputs=base_model.input, outputs=[layer.output, base_model.output])

    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Calcul des gradients par rapport aux activations de la couche convolutionnelle
    grads = tape.gradient(loss, conv_outputs)

    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0] # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la heatmap
    heatmap = tf.maximum(heatmap, 0) # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap) # Normalisation entre 0 et 1
    heatmap = heatmap.numpy() # Convertir en tableau numpy pour la visualisation

    # Redimensionner la heatmap pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (img.shape[1], img.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # Supprimer la dimension de taille 1 à la fin du tableau

    # Colorier la heatmap avec une palette (par exemple "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B

    # Superposer la heatmap à l'image
    superposed_img = heatmap_colored*0.4 + img / 255.0
    image_grad_cam = np.clip(superposed_img, 0, 1) # Garantit que toutes les valeurs de l'image finale se situent entre
    # 0 et 1

    return image_grad_cam

# Fonction pour sélectionner une image aléatoire d'un dossier
def get_random_image(folder):
    images = [img for img in os.listdir(folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        return None
    return os.path.join(folder, random.choice(images))


# Fonction permettant une attente afin de protéger des défauts de l'environnement
# (impossibilité d'accès à certains dossiers durant le traitement)
async def attendre(time):
    print("Début")
    await asyncio.sleep(time)  # 20 ms
    print("Fin")

st.title("Projet de Classification Radiographies Pulmonaires")
st.sidebar.title("Sommaire")
pages = ["Introduction", "Data Exploration", "Modélisation", "Analyse des résultats", "Prédictions"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] :
    st.write("### Introduction")

    # --- Temps total écoulé sur phase de test : 1min20s
    intro_general_1 = "La pandémie de COVID-19 a mis à rude épreuve les systèmes de santé mondiaux et a souligné la nécessité d’une collaboration internationale pour une meilleure gestion des crises sanitaires. La radiographie thoracique, en complément des tests biologiques, s’est révélée utile pour détecter les atteintes pulmonaires. L'émergence de l’intelligence artificielle offre désormais des perspectives prometteuses pour automatiser les analyses médicales."
    intro_general_2 = "Face aux futures pandémies, il est essentiel de développer ces technologies et de sensibiliser la population à l'importance de l’analyse des données médicales."
    intro_general_3 = "Lorsqu'un patient se présente aux urgences avec une suspicion ou une confirmation de COVID-19, un examen d’imagerie est recommandé pour évaluer l’état des poumons. La radiographie thoracique, bien que moins performante que le scanner, reste plus accessible et permet d’identifier des lésions compatibles avec une pneumonie virale."
    
    st.write(intro_general_1)
    st.write(intro_general_2)
    st.write(intro_general_3)

    # Charger l'image
    img = Image.open('./reports/figures/img_streamlit_1.jpg')
    img = img.resize((round(896/4), round(1152/4)))  # Redimensionnement à 25%
    # Affichage de l’image
    st.image(img)
          

if page == pages[1] :
    st.write("### Data Exploration")
    
    df=pd.read_csv(r'data/data_for_st/metadatas_with_url.csv')
    
    if st.checkbox("Afficher un échantillon des images") :
        # Dossiers contenant les images
        selected_images = {
            "COVID": "./data/data_for_st/images_raw/COVID-1.png",
            "Lung Opacity": "data/data_for_st/images_raw/Lung_Opacity-1.png",
            "Normal": "data/data_for_st/images_raw/NORMAL-1.png",
            "Viral Pneumonia": "data/data_for_st/images_raw/Viral_Pneumonia-1.png"
        }
        # Création de 4 colonnes
        cols = st.columns(4)  

        index = 0
        for category, image_path in selected_images.items():
            if image_path:
                with cols[index % 4]:
                    # Charger l'image
                    img = Image.open(image_path)
                    img = img.resize((149, 149))  # Redimensionnement à 50%

                    # Affichage de l’image
                    st.image(img, caption=category, use_container_width=True)
            index += 1

    if st.checkbox("Afficher un échantillon des masques") :
        # Dossiers contenant les images
        selected_masks = {
            "COVID": "./data/data_for_st/masks/COVID-1.png",
            "Lung Opacity": "data/data_for_st/masks/Lung_Opacity-1.png",
            "Normal": "data/data_for_st/masks/NORMAL-1.png",
            "Viral Pneumonia": "data/data_for_st/masks/Viral_Pneumonia-1.png"
        }
        # Création de 4 colonnes
        cols = st.columns(4)  

        index = 0
        for category, image_path in selected_masks.items():
            if image_path:
                with cols[index % 4]:
                    # Charger l'image
                    img = Image.open(image_path)
                    img = img.resize((149, 149))  # Redimensionnement à 50%

                    # Affichage de l’image
                    st.image(img, caption=category, use_container_width=True)
            index += 1

        st.write("Notre jeu de données est une bibliothèque de radiographies pulmonaires classés par catégorie de pathologie (COVID, Pneumonie Virale, Opacité Pulmonaire et Cas Normal). Nous avons à notre disposition les masques adaptés à chaque images.")
        st.write("Les images sont au format PNG 299x299 tandis que les masques font 256x256.")

    if st.checkbox("Afficher un échantillon d'images masquées") :
        # Dossiers contenant les images
        selected_masked = {
            "COVID": "./data/data_for_st/images_masked/COVID-1.png",
            "Lung Opacity": "data/data_for_st/images_masked/Lung_Opacity-1.png",
            "Normal": "data/data_for_st/images_masked/NORMAL-1.png",
            "Viral Pneumonia": "data/data_for_st/images_masked/Viral_Pneumonia-1.png"
        }
        # Création de 4 colonnes
        cols = st.columns(4)  

        index = 0
        for category, image_path in selected_masked.items():
            if image_path:
                with cols[index % 4]:
                    # Charger l'image
                    img = Image.open(image_path)
                    img = img.resize((149, 149))  # Redimensionnement à 50%

                    # Affichage de l’image
                    st.image(img, caption=category, use_container_width=True)
            index += 1

    if st.checkbox("Afficher une image qui comporte un appareil médical") :

       image_med_tool=cv2.imread(r'data/data_for_st/Lung_Opacity-5.png')
       st.image(image_med_tool, channels = "BGR", caption="Exemple d'image qui comporte un appareil médical")
    st.write("### Répartition des données")
    
    if st.checkbox("Afficher les graphiques de répartition du jeu de données") :
            # Initialisation des métadonnées
            df_meta = pd.read_csv("./data/data_for_st/metadatas_with_url.csv")
            
            # Configuration de la figure
            labels = 'NORMAL', 'OPACITY', 'COVID', 'VIRAL'
            explode = (0.2, 0, 0, 0) 

            # Création des graphiques
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Premier graphique - Camembert
            axes[0].pie(df_meta['LABEL'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            axes[0].axis('equal')  
            axes[0].set_title("Répartitions des cas dans le Dataset (en %)\n")
            
            # Deuxième graphique - Histogramme
            axes[1].bar(np.arange(4), df_meta['LABEL'].value_counts())
            axes[1].set_title("Nombre d'images présentes par pathologies\n")
            axes[1].set_xticks(np.arange(4))
            axes[1].set_xticklabels(['NORMAL', 'OPACITY', 'COVID', 'VIRAL'])
            st.pyplot(fig)


    st.write("### Origine des données")

    if st.checkbox("Afficher le graphiques de l'origine des données") :
        # Initialisation des métadonnées
        df_meta = pd.read_csv("./data/processed/metadatas_with_url.csv")
        df_meta['URL'] = df_meta['URL'].str.replace("https://", "", regex=False)

        fig, ax = plt.subplots()
        sns.histplot(df_meta, x='LABEL', hue='URL', multiple='stack', stat='percent', ax=ax)
        plt.title('Origine des données')
        plt.xlabel('Dataset')
        plt.ylabel('Pourcentage')
        plt.legend(list(df_meta['URL'].unique()), bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
        st.pyplot(fig)
    
    st.write("### Analyse de la distribution de pixels par pathologie")

    if st.checkbox("Afficher la densité des pixels des images") :
        mean_density = pd.read_csv("./data/processed/mean_density.csv")

        # Box plot avec Matplotlib
        # Création de la figure plus grande pour une meilleure lisibilité
        fig, ax = plt.subplots(figsize=(12, 6))

        # Récupération des labels uniques
        labels = mean_density['LABEL'].unique()

        # Positions pour chaque type d’image (décalage pour ne pas superposer)
        positions_full = np.arange(len(labels)) - 0.2
        positions_masked = np.arange(len(labels)) + 0.2

        # Création des boxplots
        box_full = ax.boxplot(
            [mean_density.loc[mean_density['LABEL'] == label, 'MeanD_FULL_IMAGE'] for label in labels],
            positions=positions_full,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color="orange"),  # Médiane visible
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
        )

        box_masked = ax.boxplot(
            [mean_density.loc[mean_density['LABEL'] == label, 'MeanD_LUNG_PART'] for label in labels],
            positions=positions_masked,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightcoral'),
            medianprops=dict(color="orange"),
            capprops=dict(color="black"),
            whiskerprops=dict(color="black"),
        )

        # Ajout du titre et des labels
        ax.set_title("Distribution de la densité des pixels", fontsize=14, fontweight="bold")
        ax.set_ylabel("Densité moyenne", fontsize=12)

        # Rotation des labels X pour éviter le chevauchement
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

        # Création d'une légende correcte
        legend_elements = [
            Line2D([0], [0], color="lightblue", lw=4, label="Image entière"),
            Line2D([0], [0], color="lightcoral", lw=4, label="Image masquée")
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        # Ajustement automatique pour éviter les chevauchements
        plt.tight_layout()

        # Affichage du graphique
        st.pyplot(fig)

        option = st.selectbox("Sélectionnez le type d'image", ("Non masquée", "Masquée"))
        
        # Sélection des histogrammes
        # Initialisation des histogrammes
        histograms_full_images = {
            "hist_COVID": np.array(pd.read_csv(r'data\processed\histogram_fullimg_COVID',index_col='Unnamed: 0')),
            "hist_NORMAL": np.array(pd.read_csv(r'data\processed\histogram_fullimg_NORMAL',index_col='Unnamed: 0')),
            "hist_VIRAL_PNEUMONIA": np.array(pd.read_csv(r'data\processed\histogram_fullimg_VIRAL_PNEUMONIA',index_col='Unnamed: 0')),
            "hist_LUNG_OPACITY": np.array(pd.read_csv(r'data\processed\histogram_fullimg_LUNG_OPACITY',index_col='Unnamed: 0'))
        }

        histograms_masked_images = {
            "hist_COVID": np.array(pd.read_csv(r'data\processed\histogram_maskimg_COVID',index_col='Unnamed: 0')),
            "hist_NORMAL": np.array(pd.read_csv(r'data\processed\histogram_maskimg_NORMAL',index_col='Unnamed: 0')),
            "hist_VIRAL_PNEUMONIA": np.array(pd.read_csv(r'data\processed\histogram_maskimg_VIRAL_PNEUMONIA',index_col='Unnamed: 0')),
            "hist_LUNG_OPACITY": np.array(pd.read_csv(r'data\processed\histogram_maskimg_LUNG_OPACITY',index_col='Unnamed: 0'))
        }

        # Sélection du type d'image
        if option == "Non masquée":
            histograms = histograms_full_images
            data = [mean_density['MeanD_FULL_IMAGE'][mean_density['LABEL'] == label] for label in mean_density['LABEL'].unique()]
            color = 'lightblue'
        else:
            histograms = histograms_masked_images
            data = [mean_density['MeanD_LUNG_PART'][mean_density['LABEL'] == label] for label in mean_density['LABEL'].unique()]
            color = 'lightcoral'

        # Configuration de la figure avec deux graphiques côte à côte
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Premier graphique - Box plot
        axes[0].boxplot(
            data, 
            positions=np.arange(len(mean_density['LABEL'].unique())), 
            widths=0.3, 
            patch_artist=True, 
            boxprops=dict(facecolor=color),
            labels=mean_density['LABEL'].unique()
        )

        axes[0].set_title("Densité de la densité des pixels", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Densité moyenne", fontsize=12)
        axes[0].set_xticks(np.arange(len(mean_density['LABEL'].unique())))
        axes[0].set_xticklabels(mean_density['LABEL'].unique(), rotation=45, ha="right", fontsize=10)

        # Deuxième graphique - Histogrammes des densités de pixels
     
        axes[1].plot(np.arange(0, 256), histograms['hist_COVID'], label="COVID", color='blue', linewidth=2)
        axes[1].plot(np.arange(0, 256), histograms['hist_NORMAL'], label="NORMAL", color='green', linewidth=2)
        axes[1].plot(np.arange(0, 256), histograms['hist_VIRAL_PNEUMONIA'], label="VIRAL PNEUMONIA", color='red', linewidth=2)
        axes[1].plot(np.arange(0, 256), histograms['hist_LUNG_OPACITY'], label="LUNG OPACITY", color='purple', linewidth=2)

        axes[1].set_title("Distribution de pixels par pathologie", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Densité de pixel", fontsize=12)
        axes[1].set_ylabel("Proportion de pixel", fontsize=12)
        axes[1].legend()
        axes[1].set_xlim((0,256))
        axes[1].set_ylim((0,0.012))

        # Ajustement pour éviter le chevauchement des éléments
        plt.tight_layout()

        # Affichage des graphiques dans Streamlit
        st.pyplot(fig)
        st.write('On remarque sur les images masquées que toutes les distributions de pixels des patients malades sont translatées vers la droite, indiquant une plus grande prévalence des pixels clairs.')

if page == pages[2] :
    st.write("### Modélisation")
    st.write("Nous avons mis en place des différents modèle de Machine, Deep et Transfer Learning. Les résultats obtenus sur les échantillons de test sont disponibles ci-dessous.")
    
    option = st.selectbox("Choisir le modèle", [
        "XGBoost",
        "CNN 1",
        "CNN 2",
        "VGG19",
        "Xception",
        "InceptionV3"
            ])
    
    if option == "XGBoost":
        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        confus=cv2.imread(r'reports/figures/Confusion_Matrix_XGB_OR.png')
        archi=cv2.imread(r'reports/figures/Model_Architecture/XGBoost.png')
        #axes[0].imshow(archi)
        #axes[0].set_title('Architecture de XGBoost Classifier')
        #axes[0].axis('off')
        #axes[1].imshow(confus)
        #axes[1].set_title('Matrice de Confusion du modèle')
        #axes[1].axis('off')
        #st.pyplot(fig)
        st.image(archi, channels="BGR", caption="Architecture de XGBoost Classifier")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")

        classif=np.load(r'reports/figures/results_OR_pca.npy', allow_pickle=True)          
        st.code(classif, language="plaintext")

        

    if option== "InceptionV3":
        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        confus=cv2.imread(r'reports\figures\Confusion_matrix__TL_InceptionV3_RETRAINED.png')
        archi=cv2.imread(r'reports\figures\Model_Architecture\InceptionV3.jpg')
        #axes[0].imshow(archi)
        #axes[0].set_title('Architecture de InceptionV3')
        #axes[0].axis('off')
        #axes[1].imshow(confus)
        #axes[1].set_title('Matrice de Confusion du modèle')
        #axes[1].axis('off')
        #st.pyplot(fig)
        st.image(archi, channels="BGR", caption="Architecture de InceptionV3")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")
        
        classif=np.load(r'reports\figures\Classification_report_TL_InceptionV3_RETRAINED.npy', allow_pickle=True)
        st.code(classif, language="plaintext")



    if option== "VGG19":
        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        confus=cv2.imread(r'reports\figures\Confusion_Matrix_VGG19.png')
        archi=cv2.imread(r'reports\figures\Model_Architecture\VGG19.jpg')
        #axes[0].imshow(archi)
        #axes[0].set_title('Architecture de VGG19')
        #axes[0].axis('off')
        #axes[1].imshow(confus)
        #axes[1].set_title('Matrice de Confusion du modèle')
        #axes[1].axis('off')
        #st.pyplot(fig)
        
        st.image(archi, channels="BGR", caption="Architecture de VGG19")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")

        #classif=np.load(r'reports\figures\Classification_report_TL_vgg19.npy', allow_pickle=True)
        #st.code(classif, language="plaintext")
        classif=cv2.imread(r'reports\figures\Classification_report_TL_vgg19.png')
        st.image(classif, channels="BGR")
    
    if option== "Xception":
        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        confus=cv2.imread(r'reports\figures\Confusion_Matrix_Xception.png')
        archi=cv2.imread(r'reports\figures\Model_Architecture\XCeption.png')
        archi = cv2.rotate(archi, cv2.ROTATE_90_CLOCKWISE)
        #axes[0].imshow(archi)
        #axes[0].set_title('Architecture de Xception')
        #axes[0].axis('off')
        #axes[1].imshow(confus)
        #axes[1].set_title('Matrice de Confusion du modèle')
        #axes[1].axis('off')

        #st.pyplot(fig)
        st.image(archi, channels="BGR", caption="Architecture de Xception")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")
        
        classif=np.load(r'reports\figures\Classification_report_TL_xception.npy', allow_pickle=True)
        st.code(classif, language="plaintext")
        

    if option== "CNN 1":
        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        confus=cv2.imread(r'reports\figures\Confusion_matrix__DL_Convol_2.png')
        archi=cv2.imread(r'reports\figures\Model_Architecture\CNN1.PNG')
        #axes[0].imshow(archi)
        #axes[0].set_title('Architecture de notre CNN')
        #axes[0].axis('off')
        #axes[1].imshow(confus)
        #axes[1].set_title('Matrice de Confusion du modèle')
        #axes[1].axis('off')
        #st.pyplot(fig)
        st.image(archi, channels="BGR", caption="Architecture de notre CNN")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")
        
        classif=np.load(r'reports\figures\Classification_report_DL_Convol_2.npy', allow_pickle=True)
        st.code(classif, language="plaintext")
    
    if option== "CNN 2":

        confus=cv2.imread(r'reports\figures\Matrice_Confusion_Pytorch_CNN2.jpg')
        archi=cv2.imread(r'reports\figures\Model_Architecture\CNN2.PNG')

        st.image(archi, channels="BGR", caption="Architecture de notre CNN")
        st.image(confus, channels="BGR", caption="Matrice de Confusion du modèle")
        
        classif=np.load(r'reports\figures\Classification_CNN2.npy', allow_pickle=True)
        st.code(classif, language="plaintext")
    
    if st.checkbox('Afficher la comparaison des modèles'):
        # Définition des données
        models = ['XGBoost', 'CNN 1', 'CNN 2', 'VGG19', 'XCeption', 'InceptionV3']
        accuracy = [80, 83, 87, 88, 87, 88]
        global_f1 = [79, 83, 86, 88, 87, 89]
        covid_f1 = [69, 71, 79, 80, 81, 81]
        covid_precision = [66, 65, 78, 72, 81, 78]

        # Création de la figure
        fig = plt.figure(figsize=(8, 5))  # Taille ajustée pour une meilleure lisibilité
        plt.title('Résumé des métriques de nos modèles', fontsize=14, fontweight="bold")

        # Tracé des courbes
        plt.plot(models, accuracy, '-o', label="Accuracy")
        plt.plot(models, global_f1, '-o', label="Global F1-score")
        plt.plot(models, covid_f1, '-o', label="COVID F1-score")
        plt.plot(models, covid_precision, '-o', label="COVID Precision")
        plt.plot(models, [80] * len(models), 'r--', label="Seuil de 80%")  # Ligne de référence à 80%

        # Configuration des axes
        plt.ylim((50, 100))
        plt.ylabel('Score (%)', fontsize=12)
        plt.xticks(rotation=30)  # Rotation des labels pour éviter le chevauchement

        # Ajout de la légende
        plt.legend()

        # Affichage dans Streamlit
        st.pyplot(fig)

        st.write(r"Notre choix s'est porté sur le modèle le plus consistant dans le scoring. Ainsi, Xception est le plus approprié car il obtient des scores supérieurs à 80% dans toutes les métriques d'intérêt")

if page == pages[3] :
    st.write("### Analyse des résultats")
    st.write("Ici nous analyserons les résultats obtenus sur le modèle entraîné à partir du modèle pré-entraîné **Xception**.")
    
    # COURBE D'APPRENTISSAGE
    st.write("## Courbe d'apprentissage")

    courbe_apprentissage = cv2.imread(r"reports\figures\courbe_apprentissage_xception.png")
    
    with st.expander("**Courbe d'apprentissage**"):
        st.image(courbe_apprentissage, channels="BGR")
    
    st.write("L'apprentissage s'est fait en **deux phases** :")
    st.markdown("""
                - Une première phase où nous entraînons les couches denses (les **couches de convolution** de Xceptions sont ainsi **gelées**)
                - Une deuxième phase où nous libérons l'ensemble des couches du modèle pour faire le **fine-tuning**
                """)
    st.write("""
             Comme on peut le voir, l'apprentissage est **stable lors de la première phase**, puis lors de la deuxième phase nous avons progressivement de l'**overfitting** qui s'installe au fur et à mesure de l'entraînement.
             Il peut y avoir plusieurs causes:
            """)
    st.markdown("""
                - Nous avons dégelé l'ensemble des couches de Xception d'un seul coup. Deux autres approches été possibles : soit dégeler progressivement les couches, soit ne libérer qu'une partie des couches profondes.
                - Cela pourrait être dû au manque de data augmentation et régularisation dans l'entraînement. Mais nous avons testé d'augmenter la régularisation et la data augmentation mais cela rend le modèle instable et non convergent.
                """)
    
    # MATRICE DE CONFUSION
    st.write("## Que nous apprend la matrice de confusion?")

    matrice_confusion_xception = cv2.imread(r"reports\figures\Confusion_Matrix_Xception.png", cv2.IMREAD_COLOR)
    matrice_confusion_xception = cv2.cvtColor(matrice_confusion_xception, cv2.COLOR_BGR2RGB)
    rapport_classification_xception = np.load(r"reports\figures\Classification_report_TL_xception.npy", allow_pickle=True)

    with st.expander("**Matrice de confusion**"):
        st.image(matrice_confusion_xception)

    with st.expander("**Rapport de classification**"):
        st.code(rapport_classification_xception, language="plaintext")

    st.write("""
             La classe **Pneumonie Viral** est très bien détectée : **95% de F1-score**.
             Sur la matrice de confusion, on voit que les classe **normale** et **opacité pulmonaire** sont les plus confondus. Il serait intéressant de visualiser un échantillon de patients "normaux" ayant été détectés comme ayant une opacité pulmonaire.
             """)
    
    echantillon = cv2.imread(r"reports\figures\bias_prediction1.png")
    zoom = cv2.imread(r"reports\figures\bias_prediction1_2.png")
    
    # ECHANTILLON D'IMAGES
    with st.expander("**Echantillon de 'NORMAL' détectés comme 'LUNG OPACITY'**"):
        st.image(echantillon, channels="BGR")
        st.image(zoom, channels="BGR")
    st.write("""
             Nous remarquons que parmis les 73 radiologies sans pathologies qui sont détectées comme opacité pulmonaire, **50% environs** comportent un **appareil médical**.
             C'est une observation intéressante puisque ça introduit une **perspective d'amélioration** de notre modèle. En effet nous observons un certain nombre d'images avec matériel médical dans notre jeu de données, mais ces images ne sont pas présentes en assez grand nombre pour permettre au modèle d'apprendre à en faire abstraction dans sa prédiction.
             Une potentielle amélioration consistrerait donc soit à trouver davantage d'images avec matériel médical, soit d'augmenter artificiellement son nombre.
             """)
    
    echantillon2 = cv2.imread(r"reports\figures\bias_prediction2_TCOVID_PNORMAL.png")
    with st.expander("**Echantillon de 'COVID' détectés comme 'NORMAL'**"):
        st.image(echantillon2, channels="BGR")
    st.write("""
             Nous avions également un certain nombre (45) de cas de **Covid-19 qui n'étaient pas détectés**. C'est une métrique importante dans notre étude car nous voulons éviter de passer à côté d'une pathologie.
             Sur l'échantillon ci-dessous, on observe des cas de Covid-19 détectés comme n'ayant pas de pathologie. 
             Il nous semble sur ces images qu'environs **50%** de celles-ci présentent des variations de **contrastes annormalement faibles**, aussi bien dans les niveaux de gris, que dans les niveaux de blancs.
             Il est possible donc que le modèle n'arrive pas à capter les motifs caractéristiques des pathologies lorsque l'image n'est pas assez contrastée.
             """)
    
    # INTERPRETABILITE
    st.write("## Interprétabilité")
    gradcam = cv2.imread(r"reports\figures\gradcam_Xception.png")
    gradcam_intermediaire = cv2.imread(r"reports\figures\gradcam_Xception_intermediaire.png")
    gradcam_comparaison = cv2.imread(r"reports\figures\gradcam_Xception_vs_VGG19.png")

    with st.expander("**GRADCAM Xception - dernière couche de convolution**"):
        st.image(gradcam, channels="BGR")

    st.write("""
             Nous pouvons observer sur l'image ci-dessus la **zone d'intérêt** du modèle sur sa dernière couche de convolution, grâce à l'algorithme GRADCAM.
             On peut voir que les zones d'intérêt sont plutôt pertinentes. En effet elles se concentrent sur les zones du poumons, et régulièrement sur le **centre**, la **zone intersticielle entre les poumons et le coeur** (non présent sur les images masquées).
             C'est très souvent sur cette zone que les opacité liées aux pathologies sont présentes.
             """)
    
    with st.expander("**GRADCAM Xception - couche intermédiaire**"):
        st.image(gradcam_intermediaire, channels="BGR")

    st.write("""
            Sur l'image ci-dessus nous observons pour les mêmes images les zones d'intérêt sur des **couches de convolution moins profondes**. Cela montre que le modèle "parcours" les poumons avant d'arriver sur la zone définitive.
            """)

    with st.expander("**Comparaison GRADCAM Xception vs Vgg19**"):
        st.image(gradcam_comparaison, channels="BGR")

    st.write("""""")
if page == pages[4]:
    st.write("### Prédictions")

    #--- Tentative de débuggage sans succès version de Keras ? -> maj OK :(

    # Charger le modèle en déclarant les objets personnalisés
    custom_objects = {"RandomRotation": RandomRotation}

    # Ouverture du modèle
    model = load_model("./models/model_xception.keras", custom_objects=custom_objects)

    base_model = model.get_layer("xception")

    class_names = ['COVID', 'LUNG OPACITY', 'NORMAL', 'VIRAL PNEUMONIA']

    uploaded_file = st.file_uploader("Choisissez un fichier", type=["png"])
    if uploaded_file is not None:
        st.write("Image choisie:", uploaded_file.name)

        with st.expander("Data augmentation"):
            rflip = st.checkbox('Random flip horizontal')
            rrotation = st.slider("Random rotation (%)", 0, 50)/100
            rzoom = st.slider("Random zoom (%)", 0, 100)/100
            rcontrast = st.slider("Random contrast (%)", 0, 100)/100

        # Conversion en image cv2
        bytes_data = uploaded_file.read()
        img = cv2.imdecode(np.asarray(bytearray(bytes_data), np.uint8), cv2.IMREAD_COLOR_BGR)

        # Preprocessing  
        img = np.expand_dims(img, axis=0)
        img = RandomFlip("horizontal")(img) if rflip else img
        img = RandomRotation(rrotation, fill_mode='constant', fill_value=0)(img)
        img = RandomZoom(rzoom, fill_mode='constant', fill_value=0)(img)
        img = RandomContrast(rcontrast)(img)
        img = img.numpy().astype(np.uint8)
        
        # Prédiction
        prediction = np.argmax(model.predict(img), axis=1)
        prediction = class_names[prediction[0]]

        col1, col2 = st.columns(2)
        # Affichage de l'image originale
        with col1:
            st.image(img, caption = f"Classe prédite : {prediction}")

        conv_layers = [layer.name for layer in base_model.layers if (isinstance(layer, Conv2D) or isinstance(layer, SeparableConv2D))][-1]
        grad_cam_image = grad_cam(img, model, base_model, conv_layers)

        with col2:
            st.image(grad_cam_image, caption = f"gradcam")
