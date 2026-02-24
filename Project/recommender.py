import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load artifacts
df = pickle.load(open("model/recipes_df.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
tfidf_matrix = pickle.load(open("model/tfidf_matrix.pkl", "rb"))

def parse_c_format(text):
    if not text or text == "character(0)":
        return []
    text = str(text)
    if text.startswith("c("):
        text = text[2:-1]
    return re.findall(r'"(.*?)"', text)

def get_first_image(image_field):
    images = parse_c_format(image_field)
    return images[0] if images else None

def recommend_recipes(predicted_ingredients, calorie_limit, top_n=5):
    query = " ".join(predicted_ingredients)
    q_vec = vectorizer.transform([query])

    similarity = cosine_similarity(q_vec, tfidf_matrix).flatten()

    temp = df.copy()
    temp["similarity"] = similarity
    temp = temp[temp["similarity"] > 0]
    temp = temp[temp["Calories"] <= calorie_limit]

    temp = temp.sort_values(
        by=["similarity", "Calories"],
        ascending=[False, True]
    )

    results = []
    for _, row in temp.head(top_n).iterrows():
        results.append({
            "name": row["Name"],
            "ingredients": row["ingredients_with_quantity"],
            "instructions": row["instruction_steps"],
            "calories": row["Calories"],
            "image": get_first_image(row["Images"])
        })

    return results
