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
 
# Vectorizer
cv = CountVectorizer(max_features=1500, stop_words="english")
vectors = cv.fit_transform(movies["tags"])
 
# ----------------------------
# TMDB API Key (hardcoded for now)
# ----------------------------
API_KEY = "9289375bcb5071be85178eb46b8afe1a"   # replace with your real TMDB key
 
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
    if not movie:
        return [], []
 
    movie_clean = movie.strip().lower()
 
    # Case-insensitive match
    matches = movies[movies["title"].str.lower() == movie_clean]
 
    if matches.empty:
        print("❌ Movie not found in dataset:", movie)
        return [], []
 
    index = matches.index[0]
 
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
        print("🎯 Selected movie from UI:", movie)
 
        names, posters = recommend(movie)
        recommendations = list(zip(names, posters))
 
        print("📦 Sending to UI:", recommendations)
 
    return render_template(
        "index.html",
        movie_names=movie_names,
        recommendations=recommendations
    )
 
 
# ----------------------------
# Run (Render compatible)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
