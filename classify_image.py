from tensorflow.keras.models import load_model

from src.features.get_absolute_path import get_absolute_path

from src.models import predict_model


model = load_model(get_absolute_path(r"models\model_xception.keras"))
class_names = ['COVID', 'LUNG OPACITY', 'NORMAL', 'VIRAL PNEUMONIA']

prediction = predict_model.predict_one_image(model, class_names)
predict_model.show_predictions(prediction, one_image=True)