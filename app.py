from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# ✅ Load your cleaned dataset here
# Make sure your file exists inside /project/src/
try:
    df = pd.read_csv("data/synthetic_data_clean.csv")   # <-- Update path if needed

    # ✅ Compute stats dictionary for dashboard
    stats = {
        "total_records": int(df.shape[0]),
        "unique_areas": int(df["area_name"].nunique()),
        "avg_density": round(df["informal_employment_density"].mean(), 2),
        "avg_nightlight": round(df["nightlight_intensity"].mean(), 2)
    }
except Exception as e:
    print("ERROR LOADING DATA:", e)
    stats = {}  # send empty dict so it doesn't break

@app.route("/")
def home():
    # ✅ Ensure stats is always passed to template
    return render_template("dashboard.html", stats=stats)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
