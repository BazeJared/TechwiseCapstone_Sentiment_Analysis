from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 .,!?\'\n]", "", text)
    return text.lower().strip()

LABELS = {
    0: "Not professional enough",
    1: "Somewhat professional",
    2: "Highly professional"
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_text = request.form["email_text"]
        cleaned = clean_text(input_text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        result = LABELS.get(pred, "Unknown")
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)