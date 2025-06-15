import pandas as pd

def preprocess_input(df, scaler):
    # Перетворення EmploymentType на числове
    df['EmploymentType'] = df['EmploymentType'].map({'Remote': 0, 'Hybrid': 1})

    # Заповнення пропусків (Заповнення відсутніх значень нулями (або можна інакше))
    df = df.fillna(0)

    # Масштабовані ознаки для кластеризації та продуктивності
    columns_to_scale = [
        'HoursWorkedPerWeek',
        'ProductivityScore',
        'WellBeingScore',
        'JobSatisfaction',
        'EnvironmentSatisfaction',
        'WorkLifeBalance',
        'EmploymentType'
    ]

    # Масштабуємо дані (StandardScaler)
    X_scaled = scaler.transform(df[columns_to_scale])

    # Розрахунок середнього добробуту для attrition
    wellbeing_level = df[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1).values.reshape(-1, 1)

    return X_scaled, wellbeing_level
