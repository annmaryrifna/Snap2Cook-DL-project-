import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pickle
import re
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# Load CNN model
# --------------------------------------------------
cnn_model = load_model("model/resnet50_fruit_veg4.keras")

with open("model/class_indices4.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# --------------------------------------------------
# Load recipe recommendation models
# --------------------------------------------------
recipes_df = pickle.load(open("model/recipes_df1.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer1.pkl", "rb"))
tfidf_matrix = pickle.load(open("model/tfidf_matrix1.pkl", "rb"))

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def parse_c_format(text):
    if not text or text == "character(0)":
        return []
    text = str(text)
    if text.startswith("c("):
        text = text[2:-1]
    return re.findall(r'"(.*?)"', text)

def get_first_image(img_field):
    imgs = parse_c_format(img_field)
    return imgs[0] if imgs else None

# --------------------------------------------------
# CNN prediction
# --------------------------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    preds = cnn_model.predict(img_arr)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return idx_to_class[idx], confidence

# --------------------------------------------------
# Recipe recommendation
# --------------------------------------------------
def recommend_recipes(predicted_ingredient, top_n=5):
    q_vec = vectorizer.transform([predicted_ingredient.lower()])
    similarity = cosine_similarity(q_vec, tfidf_matrix).flatten()

    temp = recipes_df.copy()
    temp["similarity"] = similarity
    temp = temp[temp["similarity"] > 0]
    temp = temp.sort_values(by="similarity", ascending=False)

    results = []
    for _, row in temp.head(top_n).iterrows():
       results.append({
    "name": row["Name"],
    "ingredients_simple": row["ingredient_list"],  
    "ingredients_full": row["ingredients_with_quantity"],  
    "instructions": row["instruction_steps"],
    "calories": row["Calories"],
    "image": get_first_image(row["Images"])
})


    return results

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    recipes = []

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            prediction, confidence = predict_image(image_path)
            recipes = recommend_recipes(prediction, top_n=5)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        recipes=recipes
    )

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
