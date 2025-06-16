import pandas as pd

def preprocess_input(data: dict, scaler, scaler_attrition):
    """
    Приймає словник даних одного працівника та повертає:
    - X_scaled: для кластеризації / продуктивності
    - attrition_scaled: для моделі ризику звільнення
    """

    # 🟢 Створюємо DataFrame з даних
    df = pd.DataFrame([data])

    # Явно перетворюємо числові поля
    numeric_columns = [
        'HoursWorkedPerWeek',
        'WellBeingScore',
        'JobSatisfaction',
        'EnvironmentSatisfaction',
        'WorkLifeBalance',
        'MonthlyIncome',
        'Age'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Категоріальні перетворення (наприклад, перетворення EmploymentType на числове)
    df['EmploymentType'] = df['EmploymentType'].map({'Remote': 0, 'In-Office': 1}).fillna(1)
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)

    df = df.fillna(0)

    # === Ознаки для scaler (кластеризація / продуктивність) ===
    columns_to_scale = [
        'HoursWorkedPerWeek',
        'WellBeingScore',
        'JobSatisfaction',
        'EnvironmentSatisfaction',
        'WorkLifeBalance',
        'EmploymentType'
    ]
    # Масштабуємо дані (StandardScaler)
    X_scaled = scaler.transform(df[columns_to_scale])

    # === Індекс добробуту для моделі ризику звільнення (attrition) ===
    df['WellBeingLevel'] = df[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)
    attrition_features = ['WellBeingLevel', 'OverTime', 'MonthlyIncome', 'Age']
    attrition_scaled = scaler_attrition.transform(df[attrition_features])

    return X_scaled, attrition_scaled
