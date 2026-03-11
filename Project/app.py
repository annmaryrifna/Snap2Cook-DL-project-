
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pickle
import re
import numpy as np
import mysql.connector
import bcrypt

from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "snap2cook_secret"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# MySQL Connection
# --------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",     # change if MySQL has password
        database="Snap2Cook"
    )


# --------------------------------------------------
# Load CNN Model
# --------------------------------------------------
cnn_model = load_model("model/resnet50_fruit_veg4.keras")

with open("model/class_indices4.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}


# --------------------------------------------------
# Load Recipe Recommendation Models
# --------------------------------------------------
recipes_df   = pickle.load(open("model/recipes_df2.pkl", "rb"))
vectorizer   = pickle.load(open("model/tfidf_vectorizer2.pkl", "rb"))
tfidf_matrix = pickle.load(open("model/tfidf_matrix2.pkl", "rb"))


# --------------------------------------------------
# Helper Functions
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
# CNN Prediction
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
# Recipe Recommendation
# --------------------------------------------------
def recommend_recipes(predicted_ingredient, calorie_limit=None, top_n=50):

    q_vec = vectorizer.transform([predicted_ingredient.lower()])
    similarity = cosine_similarity(q_vec, tfidf_matrix).flatten()

    temp = recipes_df.copy()
    temp["similarity"] = similarity
    temp = temp[temp["similarity"] > 0]
    temp = temp.sort_values(by="similarity", ascending=False)

    results = []

    for _, row in temp.head(top_n).iterrows():

        cal = row["Calories"]

        try:
            cal_float = float(cal) if cal else None
        except:
            cal_float = None

        if calorie_limit and cal_float and cal_float > calorie_limit:
            continue
        
        # simple ingredients (comma separated)
        ingr_simple = row["ingredient_list"]
        if isinstance(ingr_simple, list):
            ingr_simple_list = ingr_simple
        else:
            ingr_simple_list = [i.strip() for i in str(ingr_simple).split(",") if i.strip()]
            
        ingr_full = row["ingredients_with_quantity"]
        if isinstance(ingr_full, list):
            ingr_list = ingr_full
        else:
            ingr_list = [i.strip() for i in str(ingr_full).split(",") if i.strip()]

        instr = row["instruction_steps"]
        if isinstance(instr, list):
            instr_list = instr
        else:
            instr_list = [s.strip() for s in re.split(r'\n|\r\n', str(instr)) if s.strip()]

        results.append({
    "name": row["Name"],
    "ingredients_simple": ingr_simple_list,
    "ingredients_full": ingr_list,
    "instructions": instr_list,
    "calories": cal_float,
    "image": get_first_image(row["Images"])
})

    return results


# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# --------------------------------------------------
# LOGIN
# --------------------------------------------------
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):

            session["user_id"] = user["id"]
            session["username"] = user["username"]

            flash("Login successful!", "success")
            return redirect(url_for("userhome"))

        else:
            flash("Invalid username or password", "error")

    return render_template("login.html", tab="login")


# --------------------------------------------------
# SIGNUP
# --------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():

    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=%s OR email=%s",
        (username,email)
    )

    if cursor.fetchone():

        flash("User already exists","error")

        cursor.close()
        conn.close()

        return redirect(url_for("login"))

    cursor.execute(
        "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
        (username,email,hashed_pw)
    )

    conn.commit()

    cursor.close()
    conn.close()

    flash("Account created successfully! Please login.","success")

    return redirect(url_for("login"))


# --------------------------------------------------
# USER HOME
# --------------------------------------------------
@app.route("/userhome", methods=["GET", "POST"])
def userhome():
    # Check if user is logged in
    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))
    
    prediction    = None
    confidence    = None
    image_path    = None
    recipes       = []
    calorie_limit = None
    total_recipes = 0

    if request.method == "POST":
        file          = request.files.get("image")
        calorie_limit = request.form.get("calories", type=float)

        if file and file.filename:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            prediction, confidence = predict_image(img_path)
            recipes = recommend_recipes(
                prediction,
                calorie_limit=calorie_limit,
                top_n=50
            )
            total_recipes = len(recipes)
            image_path = "uploads/" + file.filename

    return render_template(
        "userhome.html",
        username      = session["username"],
        prediction    = prediction,
        confidence    = confidence,
        image_path    = image_path,
        recipes       = recipes,
        total_recipes = total_recipes,
        calorie_limit = calorie_limit,
    )

# --------------------------------------------------
# UPDATE USERNAME
# --------------------------------------------------
@app.route("/update_username", methods=["POST"])
def update_username():

    if "user_id" not in session:
        return redirect(url_for("login"))

    new_username = request.form["new_username"]
    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET username=%s WHERE id=%s",
            (new_username, user_id)
        )
        conn.commit()
        session["username"] = new_username
        flash("Username updated successfully!", "success")
    except:
        flash("Username already exists!", "error")

    cursor.close()
    conn.close()

    return redirect(url_for("userhome"))

# --------------------------------------------------
# USER Change Password
# --------------------------------------------------
@app.route("/change_password", methods=["POST"])
def change_password():

    if "user_id" not in session:
        return redirect(url_for("login"))

    current_password = request.form["current_password"]
    new_password = request.form["new_password"]

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT password FROM users WHERE id=%s",
        (user_id,)
    )

    user = cursor.fetchone()

    if not bcrypt.checkpw(current_password.encode(), user["password"].encode()):
        flash("Current password is incorrect", "error")

        cursor.close()
        conn.close()

        return redirect(url_for("userhome"))

    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

    cursor.execute(
        "UPDATE users SET password=%s WHERE id=%s",
        (hashed_pw, user_id)
    )

    conn.commit()

    cursor.close()
    conn.close()

    flash("Password changed successfully!", "success")

    return redirect(url_for("userhome"))
# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully","info")

    return redirect(url_for("login"))


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)