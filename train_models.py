# train_models.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

# Створення папки для моделей
os.makedirs("model", exist_ok=True)

# === 1. Завантаження даних ===
df_remote = pd.read_csv("remote-work-productivity.csv", delimiter=";")
df_ibm = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Перевірка розподіл ProductivityScore
print(df_remote['ProductivityScore'].describe())
print(df_remote['ProductivityScore'].value_counts(bins=5))

# === 2. Обробка remote dataset ===

# Кодуємо тип зайнятості
df_remote['EmploymentType'] = df_remote['EmploymentType'].map({'Remote': 0, 'In-Office': 1})

# Генерація ознак на основі добробуту та продуктивності
def weighted_scale(wellbeing, productivity):
    combined = (0.6 * wellbeing + 0.4 * productivity) / 100
    score = round(combined * 3 + 1 + np.random.normal(0, 0.3))
    return int(np.clip(score, 1, 4))

df_remote['JobSatisfaction'] = df_remote.apply(
    lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)

df_remote['EnvironmentSatisfaction'] = df_remote.apply(
    lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)

df_remote['WorkLifeBalance'] = df_remote.apply(
    lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)

# Визначення ознак для моделей
features_all = [
    'HoursWorkedPerWeek',
    'WellBeingScore',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'EmploymentType'
]

# Відкидаємо пропущені значення
df_remote = df_remote.dropna(subset=features_all + ['ProductivityScore'])

# Масштабування
X_all = df_remote[features_all]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Кластеризація
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Модель продуктивності
regressor = RandomForestRegressor(random_state=42)
param_grid_reg = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}
grid_reg = GridSearchCV(regressor, param_grid_reg, cv=5, scoring='r2', n_jobs=-1)
grid_reg.fit(X_scaled, df_remote['ProductivityScore'])
reg_model = grid_reg.best_estimator_

# Збереження важливості ознак
reg_feature_importance = pd.Series(reg_model.feature_importances_, index=features_all)
reg_feature_importance.sort_values(ascending=False).to_csv("model/productivity_feature_importance.csv")

# === 3. Обробка IBM HR Dataset для attrition ===

# Перетворення бінарних змінних
binary_map = {'Yes': 1, 'No': 0}
df_ibm['Attrition'] = df_ibm['Attrition'].map(binary_map)
df_ibm['OverTime'] = df_ibm['OverTime'].map(binary_map)

# Обчислення індексу добробуту
df_ibm['WellBeingLevel'] = df_ibm[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)

# Підготовка даних
features_ibm = ['WellBeingLevel', 'OverTime', 'MonthlyIncome', 'Age']
df_ibm = df_ibm.dropna(subset=features_ibm + ['Attrition'])
X_ibm = df_ibm[features_ibm]
y_ibm = df_ibm['Attrition']

# Масштабування
scaler_attrition = StandardScaler()
X_ibm_scaled = scaler_attrition.fit_transform(X_ibm)

# Класифікаційна модель
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_ibm_scaled, y_ibm)
attrition_model = grid_search.best_estimator_

# Збереження важливості ознак
feature_importances = pd.Series(attrition_model.feature_importances_, index=features_ibm)
feature_importances.sort_values(ascending=False).to_csv("model/feature_importance.csv")

# === 4. Збереження моделей ===
dump(scaler, "model/scaler.pkl")
dump(kmeans, "model/cluster_model.pkl")
dump(reg_model, "model/productivity_model.pkl")
dump(attrition_model, "model/attrition_model.pkl")
dump(scaler_attrition, "model/scaler_attrition.pkl")

print("✅ Усі моделі успішно згенеровано та збережено!")

'''
# train_models.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

# Створюємо папку для збереження моделей
os.makedirs("model", exist_ok=True)

# === 1. Завантаження даних ===
df_remote = pd.read_csv("remote-work-productivity.csv", delimiter=";")
df_ibm = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

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

# RandomForestRegressor для прогнозування ProductivtyScore
regressor = RandomForestRegressor(random_state=42)
param_grid_reg = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}
grid_reg = GridSearchCV(regressor, param_grid_reg, cv=5, scoring='r2', n_jobs=-1)
grid_reg.fit(X_scaled, df_remote['ProductivityScore'])
reg_model = grid_reg.best_estimator_

# Аналіз важливості ознак для регресії
reg_feature_importance = pd.Series(reg_model.feature_importances_, index=features_all)
reg_feature_importance.sort_values(ascending=False).to_csv("model/productivity_feature_importance.csv")

# === 3. Обробка IBM HR Dataset ===

# Перетворення категоріальних змінних у числові
binary_map = {'Yes': 1, 'No': 0}
df_ibm['Attrition'] = df_ibm['Attrition'].map(binary_map)
df_ibm['OverTime'] = df_ibm['OverTime'].map(binary_map)

# Обчислення індексу добробуту
df_ibm['WellBeingLevel'] = df_ibm[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)

# Вибір ознак
features_ibm = ['WellBeingLevel', 'OverTime', 'MonthlyIncome', 'Age']
df_ibm = df_ibm.dropna(subset=features_ibm + ['Attrition'])
X_ibm = df_ibm[features_ibm]
y_ibm = df_ibm['Attrition']

# Масштабування
scaler_attrition = StandardScaler()
X_ibm_scaled = scaler_attrition.fit_transform(X_ibm)

# RandomForestClassifier з крос-валідацією
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_ibm_scaled, y_ibm)
attrition_model = grid_search.best_estimator_

# Аналіз важливості ознак для класифікації
feature_importances = pd.Series(attrition_model.feature_importances_, index=features_ibm)
feature_importances.sort_values(ascending=False).to_csv("model/feature_importance.csv")

# === 4. Збереження моделей ===
dump(scaler, "model/scaler.pkl")
dump(kmeans, "model/cluster_model.pkl")
dump(reg_model, "model/productivity_model.pkl")
dump(attrition_model, "model/attrition_model.pkl")
dump(scaler_attrition, "model/scaler_attrition.pkl")

print("✅ Усі моделі успішно збережені!")
'''

