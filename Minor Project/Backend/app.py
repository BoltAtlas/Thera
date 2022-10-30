import imp
from flask import Flask, render_template, render_template_string, request, jsonify
from chat import *
from flask_cors import CORS
from Result import *
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
