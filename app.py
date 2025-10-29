from flask import Flask, render_template, request
import google.generativeai as genai
from config import GEMINI_API_KEY

app = Flask(__name__)
genai.configure(api_key=GEMINI_API_KEY)

@app.route("/insights")
def insights():
    prompt = "Generate strategic insights for Mumbai informal employment"
    model = genai.GenerativeModel("gemini-2.0-flash-lite")  # ✅ using flash-lite as requested
    result = model.generate_content(prompt)
    return result.text

@app.route("/")
def index():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
