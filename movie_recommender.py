import pickle
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load preprocessed data ONLY
movies = pickle.load(open("movies.pkl", "rb"))

# Lightweight vectorizer
cv = CountVectorizer(max_features=1500, stop_words="english")
vectors = cv.fit_transform(movies["tags"])  # keep sparse

API_KEY = "YOUR_TMDB_API_KEY"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": API_KEY}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w342/{data['poster_path']}"
    except Exception:
        pass
    return "https://via.placeholder.com/200x300?text=No+Poster"

def recommend(movie):
    if movie not in movies["title"].values:
        return [], []

    index = movies[movies["title"] == movie].index[0]

    # Compute similarity only for selected movie
    distances = cosine_similarity(vectors[index], vectors).flatten()

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    names, posters = [], []
    for i in movie_list:
        row = movies.iloc[i[0]]
        movie_id = row["id"]          # ✅ FIX
        names.append(row["title"])    # ✅ FIX
        posters.append(fetch_poster(movie_id))

    return names, posters

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

if __name__ == "__main__":
    app.run(debug=False)
