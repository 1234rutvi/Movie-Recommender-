import os

import pickle

import requests

import pandas as pd
 
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
 
app = Flask(__name__)
 
# ----------------------------

# Load data

# ----------------------------

movies = pickle.load(open("movies.pkl", "rb"))
 
# Ensure required columns exist

movies["title"] = movies["title"].astype(str)

movies["tags"] = movies["tags"].astype(str)
 
# Vectorizer

cv = CountVectorizer(max_features=1500, stop_words="english")

vectors = cv.fit_transform(movies["tags"])
 
# Hardcoded API key for now

API_KEY = "9289375bcb5071be85178eb46b8afe1a"  # replace with real one
 
# ----------------------------

# Helpers

# ----------------------------

def fetch_poster(movie_id):

    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}"

        params = {"api_key": API_KEY}

        r = requests.get(url, params=params, timeout=5)

        data = r.json()

        if data.get("poster_path"):

            return "https://image.tmdb.org/t/p/w342/" + data["poster_path"]

    except Exception:

        pass

    return "https://via.placeholder.com/200x300?text=No+Poster"
 
 
def recommend(movie):

    if not movie:

        return [], []
 
    # Normalize

    movie_clean = movie.strip().lower()

    titles_lower = movies["title"].str.strip().str.lower()
 
    if movie_clean not in titles_lower.values:

        print("❌ Movie not found in dataset:", movie)

        return [], []
 
    index = titles_lower[titles_lower == movie_clean].index[0]
 
    # Compute similarity

    distances = cosine_similarity(vectors[index], vectors).flatten()
 
    # If all similarities are zero (except itself), fallback

    if distances.sum() == 0:

        print("⚠️ All similarities are zero for:", movie)

        # Fallback: pick next 5 movies by index (skip itself)

        candidate_indices = [i for i in range(len(movies)) if i != index][:5]

    else:

        # Normal path

        candidate_indices = [

            i for i, _ in sorted(

                list(enumerate(distances)),

                key=lambda x: x[1],

                reverse=True

            )

            if i != index

        ][:5]
 
    names, posters = [], []
 
    for idx in candidate_indices:

        row = movies.iloc[int(idx)]
 
        movie_id = row.get("id")

        title = row.get("title")
 
        if pd.isna(movie_id) or pd.isna(title):

            continue
 
        names.append(title)

        posters.append(fetch_poster(int(movie_id)))
 
    print("✅ Recommended:", names)

    return names, posters

 
# ----------------------------

# Routes

# ----------------------------

@app.route("/", methods=["GET", "POST"])

def home():

    movie_names = sorted(movies["title"].values)

    recommendations = []
 
    if request.method == "POST":

        movie = request.form.get("movie")

        print("🎯 Selected movie:", movie)
 
        names, posters = recommend(movie)

        recommendations = list(zip(names, posters))
 
        print("📦 Sending to UI:", recommendations)
 
    return render_template(

        "index.html",

        movie_names=movie_names,

        recommendations=recommendations

    )
 
 
# ----------------------------

# Run

# ----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)

 
