from flask import Flask, render_template, request
import os

from search import load_resources, build_corpus_matrix, topk_dot

app = Flask(__name__)

# Paths (swap embeddings/word2idx to whichever model you want)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
EMB_PATH    = os.path.join(DATA_DIR, "embeddings.npy")
W2I_PATH    = os.path.join(DATA_DIR, "word2idx.json")

# Load once on startup
contexts, word2idx, emb_norm = load_resources(CORPUS_PATH, EMB_PATH, W2I_PATH)
C = build_corpus_matrix(contexts, word2idx, emb_norm)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        results = topk_dot(query, contexts, C, word2idx, emb_norm, k=10)

    return render_template("index.html", query=query, results=results, total=len(contexts))

if __name__ == "__main__":
    # debug=True for dev; turn off for submission if you want
    app.run(host="0.0.0.0", port=5000, debug=True)
