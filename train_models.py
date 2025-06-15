# train_models.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from joblib import dump
import os

# Створюємо папку для збереження моделей
os.makedirs("model", exist_ok=True)

# === 1. Завантаження даних ===
df_remote = pd.read_csv("remote-work-productivity.csv", delimiter=";")
df_ibm = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#print("Колонки remote-work-productivity.csv:", df_remote.columns.tolist())

# === 2. Обробка remote dataset ===

# Кодуємо тип зайнятості
df_remote['EmploymentType'] = df_remote['EmploymentType'].map({'Remote': 0, 'In-Office': 1})

# Додаємо колонки з IBM до df_remote (для тренування scaler)
df_remote['JobSatisfaction'] = df_ibm['JobSatisfaction'].fillna(0)
df_remote['EnvironmentSatisfaction'] = df_ibm['EnvironmentSatisfaction'].fillna(0)
df_remote['WorkLifeBalance'] = df_ibm['WorkLifeBalance'].fillna(0)

# Визначаємо всі потрібні ознаки
features_all = [
    'HoursWorkedPerWeek',
    'ProductivityScore',
    'WellBeingScore',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'EmploymentType'
]

# Відкидаємо пропущені значення
df_remote = df_remote.dropna(subset=features_all)

# Масштабування
X_all = df_remote[features_all]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Кластеризація
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Модель продуктивності (передбачення ProductivityScore)
reg_model = LinearRegression()
reg_model.fit(X_scaled, df_remote['ProductivityScore'])

# === 3. Обробка IBM HR Dataset ===

df_ibm['WellBeingLevel'] = df_ibm[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)
df_ibm = df_ibm.dropna(subset=['WellBeingLevel', 'Attrition'])

df_ibm['AttritionFlag'] = df_ibm['Attrition'].map({'Yes': 1, 'No': 0})
X_ibm = df_ibm[['WellBeingLevel']]
y_ibm = df_ibm['AttritionFlag']

log_reg = LogisticRegression()
log_reg.fit(X_ibm, y_ibm)

# === 4. Збереження моделей ===
os.makedirs("model", exist_ok=True)
dump(scaler, "model/scaler.pkl")
dump(kmeans, "model/cluster_model.pkl")
dump(reg_model, "model/productivity_model.pkl")
dump(log_reg, "model/attrition_model.pkl")

print("✅ Усі моделі успішно збережені!")







'''
# Перевіряємо необхідні колонки для remote (Визначаємо ознаки)
required_remote_columns = [
    'HoursWorkedPerWeek',
    'WellBeingScore',
    'ProductivityScore',
    'EmploymentType'
]
# Відкидаємо пропущені значення
df_remote = df_remote.dropna(subset=required_remote_columns)

# Масштабовані ознаки
X_remote = df_remote[['HoursWorkedPerWeek', 'WellBeingScore', 'EmploymentType']]
y_productivity = df_remote['ProductivityScore']

# Масштабуємо
scaler = StandardScaler()
X_remote_scaled = scaler.fit_transform(X_remote)

# === 3. Кластеризація (для типу працівника) ===
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_remote_scaled)

# === 4. Лінійна регресія для ProductivtyScore (Модель продуктивності) ===
reg_model = LinearRegression()
reg_model.fit(X_remote_scaled, y_productivity)

# === 5. Обробка IBM HR Analytics ===
# Створюємо WellBeingLevel
df_ibm['WellBeingLevel'] = df_ibm[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)
df_ibm = df_ibm.dropna(subset=['WellBeingLevel', 'Attrition'])

# Цільова змінна
df_ibm['AttritionFlag'] = df_ibm['Attrition'].map({'Yes': 1, 'No': 0})
X_ibm = df_ibm[['WellBeingLevel']]
y_ibm = df_ibm['AttritionFlag']

log_reg = LogisticRegression()
log_reg.fit(X_ibm, y_ibm)

# === 6. Збереження моделей ===
dump(scaler, "model/scaler.pkl")
dump(kmeans, "model/cluster_model.pkl")
dump(reg_model, "model/productivity_model.pkl")
dump(log_reg, "model/attrition_model.pkl")

print("✅ Усі моделі успішно збережені!")


'''

