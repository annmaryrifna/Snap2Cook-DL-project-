import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "model/resnet50_fruit_veg_finetuned.keras"
)

# Load class labels
class_indices = json.load(open("model/class_indices.json"))
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_ingredient(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    predicted_index = np.argmax(preds)

    return idx_to_class[predicted_index]
