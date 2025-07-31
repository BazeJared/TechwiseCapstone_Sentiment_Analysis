import joblib
import re
import os

if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    print("❌ ERROR: model.pkl and/or vectorizer.pkl not found in the current directory.")
    exit()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 .,!?\'\n]", "", text)
    return text.lower().strip()

LABELS = {
    0: "Not professional enough",
    1: "Somewhat professional",
    2: "Highly professional"
}

while True:
    print("\nEnter your email text (or type 'exit' to quit):")
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    print(f"\n🔎 Professionalism Rating: {LABELS[pred]}")