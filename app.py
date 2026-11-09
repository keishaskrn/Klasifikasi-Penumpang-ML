import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "decision_tree_titanic.joblib")
app = Flask(__name__)

def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            model = get_model()
            if model is None:
                raise FileNotFoundError("Model belum tersedia. Jalankan train_model.py terlebih dahulu.")
            data = pd.DataFrame([{
                "pclass": int(request.form["pclass"]),
                "sex": request.form["sex"],
                "age": float(request.form["age"]),
                "fare": float(request.form["fare"])
            }])
            result = int(model.predict(data)[0])
            prediction = "Selamat" if result == 1 else "Tidak Selamat"
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
