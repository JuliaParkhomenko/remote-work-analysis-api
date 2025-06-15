from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from utils.preprocessing import preprocess_input
from utils.recommend import get_recommendations

app = Flask(__name__)

# Завантаження моделей
cluster_model = load('model/cluster_model.pkl')
attrition_model = load('model/attrition_model.pkl')
scaler = load('model/scaler.pkl')
productivity_model = load('model/productivity_model.pkl')

@app.route("/")
def home():
    return "✅ API працює. Надішліть POST-запит на /analyze з JSON-даними."

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    df = pd.DataFrame([data])

    # Масштабуємо ознаки для кластеру і продуктивності (Попередня обробка (масштабування, кодування тощо))
    X_scaled, wellbeing_level = preprocess_input(df, scaler)

    # Прогнози
    cluster = int(cluster_model.predict(X_scaled)[0])
    predicted_productivity = float(productivity_model.predict(X_scaled)[0])
    attrition_prob = float(attrition_model.predict_proba(wellbeing_level)[0][1])

    # Рекомендації та пояснення
    recommendations, explanation = get_recommendations(cluster, attrition_prob, predicted_productivity)

    return jsonify({
        "cluster": cluster,
        "attrition_risk": round(attrition_prob, 2),
        "predicted_productivity": round(predicted_productivity, 1),
        "recommendations": recommendations,
        "explanation": explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
