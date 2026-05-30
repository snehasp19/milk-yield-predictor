from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/milk_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    parity = float(request.form["parity"])
    dim = float(request.form["days_in_milk"])
    bcs = float(request.form["bcs"])
    feed = float(request.form["feed_intake"])

    features = np.array([[parity, dim, bcs, feed]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Milk Yield: {prediction:.2f} Litres"
    )

if __name__ == "__main__":
    app.run(debug=True)