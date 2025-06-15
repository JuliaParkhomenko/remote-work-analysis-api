'''
# recommend.py

import numpy as np
from joblib import load

# Завантаження моделей
scaler = load("model/scaler.pkl")
cluster_model = load("model/cluster_model.pkl")
attrition_model = load("model/attrition_model.pkl")
productivity_model = load("model/productivity_model.pkl")

def get_recommendations(data: dict) -> dict:
    """
    Генерує текстові рекомендації на основі даних працівника.
    """
    # Отримуємо вхідні значення
    hours = data["HoursWorkedPerWeek"]
    wellbeing = data["WellBeingScore"]
    satisfaction = np.mean([
        data["JobSatisfaction"],
        data["EnvironmentSatisfaction"],
        data["WorkLifeBalance"]
    ])
    emp_type = 0 if data["EmploymentType"] == "Remote" else 1

    # Масштабуємо для моделей кластеризації та прогнозування
    X_scaled = scaler.transform([[hours, wellbeing]])

    # Кластеризація
    cluster = cluster_model.predict(X_scaled)[0]

    # Ризик звільнення (Attrition)
    attrition_prob = attrition_model.predict_proba([[satisfaction]])[0][1]

    # Продуктивність
    predicted_productivity = productivity_model.predict([[hours, wellbeing]])[0]

    # === Формування рекомендацій ===
    recommendations = []

    # Продуктивність
    if predicted_productivity < 65:
        recommendations.append("🔧 Продуктивність працівника нижча за очікувану. Рекомендується провести оцінку навантаження та забезпечити кращу підтримку.")
    else:
        recommendations.append("✅ Продуктивність перебуває на задовільному рівні.")

    # Добробут
    if wellbeing < 60:
        recommendations.append("🧘 Добробут працівника низький. Рекомендується надати доступ до програм ментального здоров’я та можливість гнучкого графіку.")
    elif wellbeing > 80:
        recommendations.append("🌟 Працівник має високий рівень добробуту. Продовжуйте підтримку наявних умов.")
    else:
        recommendations.append("📊 Добробут працівника середній. Варто дослідити джерела стресу або незадоволеності.")

    # Перевантаження
    if hours > 45:
        recommendations.append("⏱️ Перевищене робоче навантаження. Рекомендується зменшити кількість годин або делегувати завдання.")
    elif hours < 30:
        recommendations.append("⚠️ Низьке навантаження. Слід перевірити залученість працівника або обсяг задач.")

    # Ризик звільнення
    if attrition_prob > 0.5:
        recommendations.append("🚨 Є високий ризик звільнення. Рекомендується провести індивідуальну зустріч для з'ясування причин.")
    else:
        recommendations.append("🟢 Ризик звільнення низький.")

    # Тип працівника (кластер)
    cluster_map = {
        0: "Типовий працівник з помірним навантаженням і продуктивністю.",
        1: "Продуктивний працівник з меншим навантаженням — варто підтримувати та не перевантажувати.",
        2: "Працівник із високим навантаженням, але низьким добробутом — ризик вигорання."
    }
    recommendations.append(f"📌 Кластер працівника: {cluster_map[cluster]}")

    return {
        "employee_id": data.get("EmployeeID", "N/A"),
        "recommendations": recommendations
    }
'''
# recommendation.py

def get_recommendations(cluster: int, attrition_prob: float, productivity: float) -> tuple:
    """
    Генерує текстові рекомендації та пояснення на основі прогнозованих показників.

    Параметри:
    - cluster: номер внутрішньої групи працівника (індикатор типу поведінки/групи)
    - attrition_prob: ймовірність звільнення (0–1)
    - productivity: прогнозований рівень продуктивності (0-100)

    Повертає:
    - recommendations: список рекомендацій для HR
    - explanation: узагальнений опис ситуації
    """

    recommendations = []

    # --- Продуктивність ---
    if productivity < 60:
        recommendations.append("Прогнозована продуктивність працівника нижча за очікувану. "
            "Це може свідчити про втому, низьку мотивацію або невідповідність задач. "
            "Рекомендується переглянути обсяг обов’язків, надати підтримку або провести індивідуальну бесіду."
        )
        prod_label = "низька"
    elif productivity > 85:
        recommendations.append(
            "Працівник демонструє високу продуктивність. "
            "Рекомендується визнати його внесок, надати можливості для кар’єрного розвитку "
            "та не перевантажувати, щоб зберегти мотивацію."
        )
        prod_label = "висока"
    else:
        recommendations.append(
            "Продуктивність працівника знаходиться в прийнятних межах. "
            "Варто підтримувати наявні умови праці, щоб зберегти стабільний рівень ефективності."
        )
        prod_label = "середня"

    # --- Ризик звільнення ---
    if attrition_prob > 0.7:
        recommendations.append(
            "Виявлено дуже високий ризик звільнення. "
            "Ймовірно, працівник незадоволений умовами праці, має проблеми з балансом 'робота-життя' "
            "або вважає свою роботу недооціненою. Необхідно оперативно провести індивідуальну розмову "
            "та розглянути можливості покращення ситуації."
        )
        attrition_label = "дуже високий"
    elif attrition_prob > 0.4:
        recommendations.append(
            "Існує помірний ризик звільнення. "
            "HR варто звернути увагу на добробут працівника, рівень задоволеності та рівень навантаження. "
            "Профілактична розмова може запобігти небажаному переходу."
        )
        attrition_label = "помірний"
    else:
        recommendations.append(
            "Ризик звільнення наразі низький. Працівник, ймовірно, задоволений умовами та добре інтегрований у команду."
        )
        attrition_label = "низький"

    # --- Тип працівника (кластер) ---
    type_map = {
        0: (
            "Цей працівник демонструє збалансовану поведінку — середній рівень навантаження, продуктивності та добробуту. "
            "Ситуація виглядає стабільною, але варто періодично перевіряти рівень мотивації."
        ),
        1: (
            "Працівник із високим рівнем продуктивності та нижчим навантаженням. "
            "Є потенціал для розвитку, важливо не допустити втрати інтересу через одноманітність або нестачу викликів."
        ),
        2: (
            "У працівника спостерігається високе навантаження та ознаки зниженого добробуту. "
            "Це може свідчити про наближення до професійного вигорання. "
            "Рекомендується зменшити обсяг задач, запропонувати відпустку або гнучкий графік."
        )
    }
    recommendations.append(type_map.get(cluster, "Тип працівника не визначено — варто переглянути вхідні дані."))

    # --- Пояснення (для виводу у UI / звіт) ---
    explanation = (
        f"Прогнозована продуктивність: {productivity:.1f} балів — {prod_label}.\n"
        f"Ймовірність звільнення: {attrition_prob:.2f} — {attrition_label} ризик.\n"
        f"{type_map.get(cluster, '')}"
    )

    return recommendations, explanation
