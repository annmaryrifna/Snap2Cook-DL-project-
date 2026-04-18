
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import json
import pickle
import re
import numpy as np
import mysql.connector
import bcrypt
import requests
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer


SPOON_API_KEY = os.getenv("SPOON_API_KEY","")

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY","fallback_secret_key")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.getenv("MAIL_DEFAULT_SENDER")    

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)
required_env = [
    "SECRET_KEY",
    "MAIL_USERNAME",
    "MAIL_PASSWORD",
    "MAIL_DEFAULT_SENDER"
]

for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"Environment variable '{var}' is not set")
# --------------------------------------------------
# MySQL Connection
# --------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",     
        database="Snap2Cook",
        autocommit=True
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
def normalize_ingredient(label):
    label = label.lower().strip()

    corrections = {
        "raddish": "radish",
        "soy beans": "soybean",
        "sweetpotato": "sweet potato",
        "chilli pepper": "chili pepper"
    }

    return corrections.get(label, label)

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

def contains_all(recipe_ingredients, predicted):
    text = " ".join(recipe_ingredients).lower()
    return all(p.lower() in text for p in predicted)

# --------------------------------------------------
# CNN Prediction
# --------------------------------------------------
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    preds = cnn_model.predict(img_arr)[0]

    # Top 3 predictions
    top_indices = preds.argsort()[-3:][::-1]

    ingredients = []

    for i in top_indices:
        ingredients.append({
            "name": idx_to_class[i],
            "confidence": float(preds[i])
        })

    return ingredients


# --------------------------------------------------
# Recipe Recommendation
# --------------------------------------------------
def recommend_recipes(predicted_ingredient, calorie_limit=None, top_n=50):

    query = " ".join(predicted_ingredient).lower()
    q_vec = vectorizer.transform([query])
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
        if len(predicted_ingredient) > 1:
            if not contains_all(ingr_simple_list, predicted_ingredient):
                continue
        results.append({
    "name": row["Name"],
    "ingredients_simple": ingr_simple_list,
    "ingredients_full": ingr_list,
    "instructions": instr_list,
    "calories": cal_float,
    "image": get_first_image(row["Images"])
})
    random.shuffle(results)
    return results

# --------------------------------------------------
# Recipe Recommendation using spoonacular API
# --------------------------------------------------
def fetch_spoonacular_recipes(ingredients, calorie_limit=None, max_results=10):
    if len(ingredients) == 1:
        query = ingredients[0]+" recipe"
    else:
        query = " ".join(ingredients)

    url = "https://api.spoonacular.com/recipes/complexSearch"

    params = {
        "query": query,
        "number": max_results,
        "addRecipeInformation": True,
        "fillIngredients": True,
        "addRecipeNutrition": True,
        "instructionsRequired": True,
        "apiKey": SPOON_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
    except:
        return []

    recipes = []

    for r in data.get("results", []):

        # ---------- INGREDIENTS ----------
        ingr_full = []
        ingr_simple = []

        for i in r.get("extendedIngredients", []):
            name = i.get("name", "")
            amount = i.get("amount", "")
            unit = i.get("unit", "")
            if isinstance(amount,float) and amount.is_integer():
                amount = int(amount)
            ingr_simple.append(name)

            if amount:
                ingr_full.append(f"{amount} {unit} {name}".strip())
            else:
                ingr_full.append(name)

        # ---------- INSTRUCTIONS ----------
        instructions = []
        if r.get("analyzedInstructions"):
            for ins in r["analyzedInstructions"]:
               for step in ins.get("steps", []):
                   if step.get("step"):
                       instructions.append(step["step"].strip())
        if not instructions:
            raw=r.get("instructions")
            if raw:
                clean = re.sub('<.*?>', '', raw)
                instructions = [s.strip() for s in clean.split("\n") if s.strip()]
        if not instructions:
            instructions=["No instructions available."]
            
        # ---------- CALORIES ----------
        calories = None
        nutrients = r.get("nutrition", {}).get("nutrients", [])

        for n in nutrients:
            if n["name"] == "Calories":
                calories = int(n["amount"])

        # ---------- FINAL OBJECT ----------
        recipes.append({
            "name": r["title"],
            "ingredients_simple": ingr_simple,
            "ingredients_full": ingr_full,
            "instructions": instructions,
            "calories": calories,
            "image": r["image"]
        })

    return recipes
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

        #username = request.form["username"]
        login_input = request.form.get("login")
        password = request.form.get("password")

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE username=%s or email=%s", (login_input, login_input,))
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
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if not username or not email or not password:
            flash("All fields are required", "error")
            return redirect(url_for("login"))

        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return redirect(url_for("login"))

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check email
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already registered.", "error")
            return redirect(url_for("login"))

        # Check username
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            flash("Username already taken.", "error")
            return redirect(url_for("login"))

        # Insert
        cursor.execute(
            "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
            (username,email,hashed_pw)
        )

        conn.commit()
        cursor.close()
        conn.close()

        flash("Account created successfully!", "success")
        return redirect(url_for("login"))

    # ✅ VERY IMPORTANT (handles GET)
    return render_template("login.html", tab="signup")# --------------------------------------------------
# USER HOME
# --------------------------------------------------
@app.route("/userhome", methods=["GET", "POST"])
def userhome():

    # Check if user is logged in
    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    image_paths = []
    recipes = []
    calorie_limit = None
    total_recipes = 0
    predictions = []

    if request.method == "POST":

        files = request.files.getlist("image")   # multiple images
        calorie_limit = request.form.get("calories", type=float)

        ingredient_list = []
        predictions = []
        image_paths = []
        for file in files:
            if file and file.filename:
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)

                img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(img_path)
                preds = predict_image(img_path)

                if preds:
                    raw_ingredient = preds[0]["name"]
                    conf = preds[0]["confidence"]
                    ingredient = normalize_ingredient(raw_ingredient)

                    predictions.append({
                "name": ingredient,
                "confidence": conf,
                "image": "uploads/" + filename
                  })

                    ingredient_list.append(ingredient)

                image_paths.append("uploads/" + filename)

        # Remove duplicates
        ingredient_list =  list(dict.fromkeys(ingredient_list))

        if predictions:
            prediction = predictions[0]["name"]
            confidence = predictions[0]["confidence"]

        recipes = recommend_recipes(
            ingredient_list,
            calorie_limit=calorie_limit,
            top_n=30
        )

        
        if len(ingredient_list) > 1 and total_recipes == 0:
            api_recipes = fetch_spoonacular_recipes( ingredient_list,calorie_limit, max_results=20)
            filtered_api = [  r for r in api_recipes
                               if contains_all(r["ingredients_simple"], ingredient_list)
                            ]
            if filtered_api:
                recipes = filtered_api
            else:
                recipes = []  

        
        if len(recipes) < 5:
            api_recipes = fetch_spoonacular_recipes(
            ingredient_list,
            calorie_limit,max_results=15  )
            
            filtered_api = [r for r in api_recipes if contains_all(r["ingredients_simple"], ingredient_list) and r["instructions"] != ["No instructions available."]]
            api_recipes = filtered_api if filtered_api else api_recipes
            random.shuffle(api_recipes)
            recipes.extend(api_recipes)
        total_recipes = len(recipes)
    return render_template(
        "userhome.html",
        username=session["username"],
        prediction=prediction,
        predictions=predictions,
        confidence=confidence,
        image_paths=image_paths,
        recipes=recipes,
        total_recipes=total_recipes,
        calorie_limit=calorie_limit,
    )

# --------------------------------------------------
# SAVE RECIPE
# --------------------------------------------------
@app.route("/save_recipe", methods=["POST"])
def save_recipe():

    if "user_id" not in session:
        return {"status": "error", "message": "Login required"}

    data = request.json

    user_id = session["user_id"]

    predicted = ", ".join(data.get("predicted", []))
    name = data.get("name")
    image = data.get("image")
    calories = data.get("calories")

    ingredients = json.dumps(data.get("ingredients", []))
    instructions = json.dumps(data.get("instructions", []))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO search_history
        (user_id, predicted_ingredient, recipe_name, recipe_image, calories, ingredients, instructions)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (user_id, predicted, name, image, calories, ingredients, instructions))

    conn.commit()
    cursor.close()
    conn.close()

    return {"status": "success"}

# --------------------------------------------------
# RECIPE HISTORY
# --------------------------------------------------
@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM search_history
        WHERE user_id = %s
        ORDER BY saved_at DESC
    """, (session["user_id"],))

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    # convert JSON back to list
    for r in data:
        r["ingredients"] = json.loads(r["ingredients"])
        r["instructions"] = json.loads(r["instructions"])

    return render_template("history.html", recipes=data, username=session["username"])

# --------------------------------------------------
# DELETE SINGLE HISTORY ENTRY
# --------------------------------------------------
@app.route("/history/delete/<int:entry_id>", methods=["POST"])
def delete_history(entry_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
 
    conn = get_db_connection()
    cursor = conn.cursor()
 
    cursor.execute(
        "DELETE FROM search_history WHERE id = %s AND user_id = %s",
        (entry_id, session["user_id"])
    )
    conn.commit()
    cursor.close()
    conn.close()
 
    flash("Recipe deleted from history.", "success")
    return redirect(url_for("history"))
 
# -------------------------------------------------- 
# CLEAR ALL HISTORY FOR THIS USER
# --------------------------------------------------
@app.route("/history/clear", methods=["POST"])
def clear_history():
    if "user_id" not in session:
        return redirect(url_for("login"))
 
    conn = get_db_connection()
    cursor = conn.cursor()
 
    cursor.execute(
        "DELETE FROM search_history WHERE user_id = %s",
        (session["user_id"],)
    )
    conn.commit()
    cursor.close()
    conn.close()
 
    flash("All history cleared.", "info")
    return redirect(url_for("history"))
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
# FORGOT PASSWORD
# --------------------------------------------------
@app.route("/forgot_password", methods=["GET","POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form["email"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:

            token = serializer.dumps(email, salt='password-reset')
            reset_url = url_for('reset_password', token=token, _external=True)

            msg = Message(
                "Snap2Cook Password Reset",
                recipients=[email]
            )

            msg.body = f"""
     Reset your password by clicking the link below:

{reset_url}

If you didn't request this, ignore this email.
"""

            mail.send(msg)

        flash("If the email exists, a reset link was sent.","info")
        return redirect(url_for("login"))
    return redirect(url_for("login"))
# --------------------------------------------------
# RESET PASSWORD
# --------------------------------------------------
@app.route("/reset-password/<token>", methods=["GET","POST"])
def reset_password(token):

    try:
        email = serializer.loads(token, salt='password-reset', max_age=900)
    except:
        flash("Invalid or expired link","error")
        return redirect(url_for("login"))

    if request.method == "POST":

        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match","error")
            return redirect(request.url)

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (hashed_pw,email)
        )

        conn.commit()
        cursor.close()
        conn.close()

        flash("Password updated successfully","success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")
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
    app.run(debug=False)