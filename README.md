<img src="https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png">
Projet - Détection du COVID et des Infections Pulmonaires

==============================

Énoncé du sujet : "Afin de faire le diagnostic des patients au Covid-19, l’analyse de radiographies pulmonaires est une possibilité à explorer pour détecter plus facilement les cas positifs. Si la classification par le biais du deep learning de telles données se révèle efficace pour détecter les cas positifs, alors cette méthode peut être utilisée dans les hôpitaux et cliniques quand on ne peut pas faire de test classique."

Élargissement de l'étude : nous avons travaillé et apporté des résultats probant sur la classification d'images médicales de type TDM concernant les pathologies suivantes : COVID19, Pneumonie Virale et Oppacité Pulmonaire (cancer). Nous détaillerons ici le procédé, le code ainsi que le modèle entraîné.

Notre équipe est constitué de : Antoine BAS, Jeremy CHOUIPPE, Antoine CARTON et Andreas LATOUR

Mentor du projet : Romain LESIEUR

------------


Lien du dataset : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

Modèle pré-entraîné : XCeption - <a href="https://drive.google.com/file/d/1fL3XmuDcn8sUlyXYqhaTur1q05RV6xkv/view?usp=sharing">Télécharger le modèle entraîné</a>

Résultat : Accuracy Globale 87%, F1-Score Global 87% --- Précision sur la classe COVID 81%, F1-Score sur la classe COVID 81%

Compte rendu : <a href="https://drive.google.com/file/d/1mZQYnfEzyF0613D-1V2dujmzGoyqaw-q/view?usp=sharing">Exploration</a> - <a href="https://drive.google.com/file/d/1Q20GiJs8xz7-4-YpmVBg4J4MS5IelPfD/view?usp=sharing">Modélisation</a> - <a href="https://drive.google.com/file/d/1tB9mLraUnCLDECImCxVcnLDZfqVDs6es/view?usp=sharing">Conclusion</a>

------------

Pour reproduire les étapes que nous avons suivi pour l'entraînement du modèle, vous trouverez le notebook main.ipynb dans la racine du projet. Celui-ci reconstruit les métadonnées, réalise le pré-traitement sur les images,
enregistre les images en local, puis ré-entraîne le modèle. Enfin il comporte une partie de prédiction si vous voulez vérifier son bon fonctionnement ou faire de nouvelles prédictions.

Ce fichier possède des dépendances disponibles dans les dossiers suivants:

- src\features\build_features.py
    
- src\models\train_model.py
    
- src\models\predict_model.py
    
Ces dépendances contiennent le code que nous avons utilisé dans notre projet pour construire le modèle final.

------------
Organisation du Projet

    ├── LICENSE
    ├── README.md          <- Le fichier README de niveau supérieur pour les développeurs utilisant ce projet.
    ├── data               <- Devrait être sur votre ordinateur mais pas sur Github (seulement dans .gitignore)
    │   ├── processed      <- Les ensembles de données finaux et canoniques pour la modélisation.
    │   └── raw            <- La source de données originale et immuable.
    │
    ├── models             <- Ce dossier est vide car le modèle a un poids suppérieur au fichier autorisé
    │
    ├── notebooks          <- Notebooks Jupyter. La convention de nommage est un numéro (pour l'ordre),
    │                         le nom du créateur et une courte description délimitée par des tirets, par exemple :
    │                         `1.0-alban-exploration-des-donnees`.
    │
    ├── references         <- Dictionnaires de données, manuels, liens et tous les autres documents explicatifs.
    │
    ├── reports            <- Les rapports que vous réaliserez pendant ce projet en PDF.
    │   └── figures        <- Graphiques et figures générés à utiliser dans les rapports.
    │
    ├── requirements.txt   <- Le fichier des dépendances pour reproduire l'environnement d'analyse, par exemple
    │                         généré avec `pip freeze > requirements.txt`.
    │
    ├── src                <- Code source à utiliser dans ce projet.
    │   ├── __init__.py    <- Rend src un module Python.
    │   │
    │   ├── features       <- Scripts pour transformer les données brutes en fonctionnalités pour la modélisation.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts pour entraîner les modèles puis utiliser les modèles entraînés pour faire des prédictions
    │   │   │                 
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts pour créer des visualisations exploratoires et orientées résultats.
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
