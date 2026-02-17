import os

import pickle

import requests

import pandas as pd
 
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
 
app = Flask(__name__)
print("🚀 App starting...")
 
# ----------------------------

# Load data

# ----------------------------

movies = pickle.load(open("movies.pkl", "rb"))
print("📊 Movies shape:", movies.shape)

print("📊 Movies columns:", movies.columns.tolist())

print("📊 Sample rows:\n", movies.head())

 
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

def fetch_poster_by_title(title):

    try:

        search_url = "https://api.themoviedb.org/3/search/movie"

        params = {

            "api_key": API_KEY,

            "query": title

        }

        r = requests.get(search_url, params=params, timeout=5)

        data = r.json()
 
        results = data.get("results")

        if results and len(results) > 0:

            poster_path = results[0].get("poster_path")

            if poster_path:

                return "https://image.tmdb.org/t/p/w342/" + poster_path

    except Exception as e:

        print("Poster fetch error:", e)
 
    return "https://via.placeholder.com/200x300?text=No+Poster"
 
def recommend(movie):

    if not movie:

        return [], []
 
    movie_clean = movie.strip().lower()

    titles_lower = movies["title"].astype(str).str.strip().str.lower()
 
    if movie_clean not in titles_lower.values:

        print("❌ Movie not found in dataset:", movie)

        print("Sample titles:", titles_lower.head().tolist())

        return [], []
 
    index = titles_lower[titles_lower == movie_clean].index[0]
 
    # Compute similarity

    distances = cosine_similarity(vectors[index], vectors).flatten()
 
    # Rank movies (even if all similarities are zero)

    ranked = sorted(

        list(enumerate(distances)),

        key=lambda x: x[1],

        reverse=True

    )
 
    # Pick top 5 excluding itself

    candidate_indices = [i for i, _ in ranked if i != index][:5]

    print("🎯 Candidate indices:", candidate_indices)
 
    names, posters = [], []
 
    for idx in candidate_indices:

        row = movies.iloc[int(idx)]
 
        title = str(row.get("title"))

        names.append(title)
 
        # Fetch poster by TITLE (not by id)

        posters.append(fetch_poster_by_title(title))
 
    print("✅ Final recommended names:", names)

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

 
