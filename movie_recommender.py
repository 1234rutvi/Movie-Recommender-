import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# ----------------- LOAD AND PROCESS DATA -----------------

import pickle

movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# Merge datasets
movies = movies.merge(credits, on="title")

# IMPORTANT: Use 'id' column for TMDB API
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# ----------------- HELPER FUNCTIONS -----------------

def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def clean_data(items):
    return [i.replace(" ", "").lower() for i in items]

# ----------------- DATA CLEANING -----------------

movies['genres'] = movies['genres'].apply(convert).apply(clean_data)
movies['keywords'] = movies['keywords'].apply(convert).apply(clean_data)
movies['cast'] = movies['cast'].apply(convert_cast).apply(clean_data)
movies['crew'] = movies['crew'].apply(fetch_director).apply(clean_data)
movies['overview'] = movies['overview'].apply(lambda x: x.lower().split())

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# ----------------- VECTORIZATION -----------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ----------------- FLASK APP -----------------

app = Flask(__name__)

# 🔑 PUT YOUR TMDB API KEY HERE
API_KEY = "b25a5793d33d4164e54869a6b9c8df22"
import time

poster_cache = {}

def fetch_poster(movie_id):
    if movie_id in poster_cache:
        return poster_cache[movie_id]

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    for _ in range(2):  # retry twice
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=8
            )

            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")

                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w342/{poster_path}"
                else:
                    poster_url = "https://via.placeholder.com/200x300?text=Poster+Not+Available"

                poster_cache[movie_id] = poster_url
                return poster_url

        except requests.exceptions.RequestException as e:
            print("Poster error:", e)
            time.sleep(1)  # wait before retry

    return "https://via.placeholder.com/200x300?text=Poster+Not+Available"


def recommend(movie):
    if movie not in movies['title'].values:
        return [], []

    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].id   # IMPORTANT FIX
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# ----------------- ROUTES -----------------

@app.route("/", methods=["GET", "POST"])
def home():
    movie_names = sorted(movies['title'].values)
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

# ----------------- RUN APP -----------------

if __name__ == "__main__":
    app.run(debug=True) 
