from flask import Flask, render_template, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mumbai_all_8_areas_data.csv")

# ✅ Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    stats = {
        "total_records": int(df.shape[0]),
        "unique_areas": int(df["area_name"].nunique()),
        "avg_density": round(df["informal_employment_density"].mean(), 2),
        "avg_nightlight": round(df["nightlight_intensity"].mean(), 2)
    }
except Exception as e:
    print(f"❌ ERROR LOADING DATA: {e}")
    stats = {}

# ROUTES
@app.route("/")
def dashboard():
    return render_template("dashboard.html", stats=stats)

@app.route("/dashboard")
def dashboard_alias():
    return render_template("dashboard.html", stats=stats)

@app.route("/predict")
def predictions():
    return render_template("predictions.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

# ✅ Serve embedded map HTML
@app.route("/assets/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets/maps"), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
