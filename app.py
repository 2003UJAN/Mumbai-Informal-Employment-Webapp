from flask import Flask, render_template, jsonify
import google.generativeai as genai
from config import GEMINI_API_KEY

app = Flask(__name__)
genai.configure(api_key=GEMINI_API_KEY)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/insights")
def insights():
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    prompt = "Generate data-driven insights for Mumbai informal employment trends."
    response = model.generate_content(prompt)
    return jsonify({"insights": response.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
