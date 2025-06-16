from flask import Flask, request, jsonify, render_template
from joblib import load
from utils.preprocessing import preprocess_input
from utils.recommend import get_recommendations
import pandas as pd
import os

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# === Завантаження моделей і скейлерів ===
scaler = load('model/scaler.pkl')
scaler_attrition = load('model/scaler_attrition.pkl')
productivity_model = load('model/productivity_model.pkl')
attrition_model = load('model/attrition_model.pkl')
cluster_model = load('model/cluster_model.pkl')


# === Головна сторінка ===
@app.route('/')
def index():
    return render_template('index.html')

# === Обробка форми аналізу працівника ===
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json

    try:
        # Попередня обробка
        X_scaled, attrition_scaled = preprocess_input(data, scaler, scaler_attrition)

        # Прогнози
        cluster = int(cluster_model.predict(X_scaled)[0])
        predicted_productivity = float(productivity_model.predict(X_scaled)[0])
        attrition_prob = float(attrition_model.predict_proba(attrition_scaled)[0][1])

        # Отримання рекомендацій
        recommendations, explanation = get_recommendations(
            cluster=cluster,
            attrition_prob=attrition_prob,
            productivity=predicted_productivity
        )

        return jsonify({
            "cluster": cluster,
            "predicted_productivity": round(predicted_productivity, 1),
            "attrition_risk": round(attrition_prob, 2),
            "recommendations": recommendations,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)