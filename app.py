from flask import Flask
app = Flask(__name__)
from flask import Flask, render_template, request
import pandas as pd
import pickle
import google.generativeai as genai
from config import GEMINI_API_KEY

app = Flask(__name__)

# Load model
model = pickle.load(open("models/mumbai_rf_model_all_areas.pkl", "rb"))

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

# Load CSV data
stats_df = pd.read_csv("data/mumbai_all_areas_statistics.csv")
pred_df = pd.read_csv("data/mumbai_all_areas_predictions.csv")


# ---- ROUTES ---- #
@app.route("/")
@app.route("/dashboard")
def dashboard():
    """Shows dashboard with charts"""
    return render_template("dashboard.html",
                           stats=stats_df.to_dict(orient="records"),
                           predictions=pred_df.to_dict(orient="records"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["nightlight_intensity"]),
                float(request.form["population_density"]),
                float(request.form["rainfall"]),
                float(request.form["temperature"]),
            ]

            prediction = model.predict([features])
            result = round(prediction[0], 2)

        except:
            result = "Error: Invalid inputs"

    return render_template("predictions.html", result=result)


@app.route("/insights")
def insights():
    """Generate AI insights from dataset summary using Gemini"""
    
    with open("data/mumbai_all_areas_summary.txt", "r") as f:
        dataset_summary = f.read()

    prompt = f"""
    You are an urban analytics expert.

    Based on this Mumbai dataset:
    {dataset_summary}

    Produce:
    ✅ Top 5 insights (bullet points)
    ✅ 3 risks
    ✅ 3 recommendations
    Use simple language.
    """

    ai_output = gemini.generate_content(prompt).text

    return render_template("insights.html", gemini_insights=ai_output)


if __name__ == "__main__":
    app.run(debug=True)
