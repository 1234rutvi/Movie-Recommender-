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
 
# Vectorizer (lightweight)

cv = CountVectorizer(max_features=1500, stop_words="english")

vectors = cv.fit_transform(movies["tags"])  # keep sparse
 
# ----------------------------

# Read API key from env (Render dashboard)

# ----------------------------

# In Render, set env var: TMDB_API_KEY = your_key_here

API_KEY ="9289375bcb5071be85178eb46b8afe1a"
 
 
# ----------------------------

# Helpers

# ----------------------------

def fetch_poster(movie_id):

    if not API_KEY:

        return "https://via.placeholder.com/200x300?text=No+API+Key"
 
    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}"

        params = {"api_key": API_KEY}

        resp = requests.get(url, params=params, timeout=5)

        data = resp.json()
 
        if data.get("poster_path"):

            return f"https://image.tmdb.org/t/p/w342/{data['poster_path']}"

    except Exception:

        pass
 
    return "https://via.placeholder.com/200x300?text=No+Poster"
 
 
def recommend(movie):

    if movie not in movies["title"].values:

        return [], []
 
    index = movies[movies["title"] == movie].index[0]
 
    # Compute similarity for selected movie

    distances = cosine_similarity(vectors[index], vectors).flatten()
 
    movie_list = sorted(

        list(enumerate(distances)),

        key=lambda x: x[1],

        reverse=True

    )[1:6]
 
    names, posters = [], []
 
    for idx, _ in movie_list:

        row = movies.iloc[int(idx)]
 
        movie_id = row.get("id")

        title = row.get("title")
 
        if pd.isna(movie_id) or pd.isna(title):

            continue
 
        names.append(title)

        posters.append(fetch_poster(int(movie_id)))
 
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

        names, posters = recommend(movie)

        recommendations = zip(names, posters)
 
    return render_template(

        "index.html",

        movie_names=movie_names,

        recommendations=recommendations

    )
 
 
# ----------------------------

# Run (Render compatible)

# ----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))  # Render provides PORT

    app.run(host="0.0.0.0", port=port, debug=False)

 

