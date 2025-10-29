from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

# ✅ Automatically build correct path (works on Render + local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mumbai_all_8_areas_data.csv")

# Try loading dataset
try:
    df = pd.read_csv(DATA_PATH)

    stats = {
        "total_records": int(df.shape[0]),
        "unique_areas": int(df["area_name"].nunique()),
        "avg_density": round(df["informal_employment_density"].mean(), 2),
        "avg_nightlight": round(df["nightlight_intensity"].mean(), 2)
    }

    print("✅ DATA LOADED SUCCESSFULLY")

except Exception as e:
    print(f"❌ ERROR LOADING DATA: {e}")
    stats = {}  # prevents crashing

@app.route("/")
def home():
    return render_template("dashboard.html", stats=stats)

@app.route("/predictions")
def predictions():
    return render_template("predictions.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
