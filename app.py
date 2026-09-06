import joblib
from preprocessor import process_review
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    reviews = data["reviews"]

    processed_review = process_review(reviews)

    processed_review = vectorizer.transform([processed_review])

    prediction = model.predict(processed_review)

    return jsonify({
        "prediction": "positive" if str(prediction[0]) == "1" else "negative"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
