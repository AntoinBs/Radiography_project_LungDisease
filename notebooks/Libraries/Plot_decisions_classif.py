import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_classification_results(model, test_ds, plot_errors=True, num_samples=3):
    """
    Plot les erreurs ou les succès de classification sur l'échantillon de test d'un modèle entraîné.
    
    Paramètres:
    - model: Le réseau de neurones convolutif entraîné.
    - test_ds: l'échantillon de test (images,labels).
    - plot_errors: Booléen, True= plot des erreurs, False= plot les succès.
    - num_samples: Nombre d'échantillons à plotter.
    """
    # Extraction des images et labels
    X = []
    Y = []
    for images, labels in test_ds:
        X.extend(images)
        Y.extend(labels)

    # Prédictions
    test_pred_model = model.predict(np.array(X))
    test_pred_class = np.argmax(test_pred_model, axis=1)
    y_test_class = np.array([label.numpy() if hasattr(label, 'numpy') else label for label in Y])

    # Détection des erreurs ou succès
    indexes = []
    for i in range(len(test_pred_class)):
        if (test_pred_class[i] != y_test_class[i]) == plot_errors:
            indexes.append(i)

    if not indexes:
        print("Aucune correspondance pour les critères choisis.")
        return

    plt.figure(figsize=(20, 8))

    # Plotter des images sélectionnées aléatoirement
    for j, i in enumerate(np.random.choice(indexes, size=min(num_samples, len(indexes)), replace=False)):
        img = X[i]
        plt.subplot(1, num_samples, j + 1)
        plt.axis('off')
        plt.imshow(img, cmap=cm.binary, interpolation='None')
        plt.title(f'True Label: {y_test_class[i]}\n'
                  f'Prediction: {test_pred_class[i]}\n'
                  f'Confidence: {round(test_pred_model[i][test_pred_class[i]], 2)}')
    
    plt.show()
