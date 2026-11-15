from flask import Flask, request, jsonify
from predict import predict_wine, FEATURES
import os

app = Flask(__name__)

label_map = {0: "class_0", 1: "class_1", 2: "class_2"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    pred, confidence = predict_wine(data)
    return jsonify({
        "prediction": label_map.get(pred, str(pred)),
        "confidence": round(confidence, 3),
    })

if __name__ == '__main__':
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )