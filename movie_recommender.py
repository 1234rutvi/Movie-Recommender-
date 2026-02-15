import pickle
import requests
import time
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- LOAD DATA -----------------

# Load preprocessed movies.pkl
movies = pickle.load(open("movies.pkl", "rb"))

# Keep features small for memory safety
cv = CountVectorizer(
    max_features=2000,
    stop_words="english"
)

# IMPORTANT: Keep sparse matrix (NO .toarray())
vectors = cv.fit_transform(movies["tags"])

# ----------------- FLASK APP -----------------

app = Flask(__name__)

API_KEY = "YOUR_TMDB_API_KEY_HERE"

poster_cache = {}

def fetch_poster(movie_id):
    if movie_id in poster_cache:
        return poster_cache[movie_id]

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=6)

        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")

            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w342/{poster_path}"
            else:
                poster_url = "https://via.placeholder.com/200x300?text=Poster+Not+Available"

            poster_cache[movie_id] = poster_url
            return poster_url

    except:
        pass

    return "https://via.placeholder.com/200x300?text=Poster+Not+Available"


# ----------------- RECOMMEND FUNCTION -----------------

def recommend(movie):

    if movie not in movies["title"].values:
        return [], []

    index = movies[movies["title"] == movie].index[0]

    # Compute similarity only for selected movie
    distances = cosine_similarity(
        vectors[index],
        vectors
    ).flatten()

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters


# ----------------- ROUTES -----------------

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


# ----------------- RUN APP -----------------

if __name__ == "__main__":
    app.run(debug=False)
