import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score

# === 1. Створення папки для моделей та графіків ===
os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

# === 2. Завантаження даних ===
df_remote = pd.read_csv("remote-work-productivity.csv", delimiter=";")
df_ibm = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# === 3. Обробка remote dataset ===
# Кодуємо тип зайнятості
df_remote['EmploymentType'] = df_remote['EmploymentType'].map({'Remote': 0, 'In-Office': 1})

# Генерація додаткових ознак на основі добробуту та продуктивності
def weighted_scale(wellbeing, productivity):
    combined = (0.6 * wellbeing + 0.4 * productivity) / 100
    score = round(combined * 3 + 1 + np.random.normal(0, 0.3))
    return int(np.clip(score, 1, 4))

for col in ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']:
    df_remote[col] = df_remote.apply(lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)

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
X_reg = df_remote[features_all]
y_reg = df_remote['ProductivityScore']
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Кластеризація
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_reg_scaled)

# Регресія (Модель продуктивності)
regressor = RandomForestRegressor(random_state=42)
param_grid_reg = {'n_estimators': [100], 'max_depth': [10, None]}
grid_reg = GridSearchCV(regressor, param_grid_reg, cv=5, scoring='r2', n_jobs=-1)
grid_reg.fit(X_train_reg, y_train_reg)
reg_model = grid_reg.best_estimator_

# Збереження важливостей ознак
reg_importance = pd.Series(reg_model.feature_importances_, index=features_all).sort_values(ascending=False)
reg_importance.to_csv("model/productivity_feature_importance.csv")

# Побудова графіка важливості
plt.figure(figsize=(8,5))
reg_importance.plot(kind='bar')
plt.title("Важливість ознак для моделі продуктивності")
plt.tight_layout()
plt.savefig("output/regression_feature_importance.png")
plt.close()


# === 4. Обробка IBM HR Dataset для класифікації (звільнення) ===
# Перетворення бінарних змінних
df_ibm['Attrition'] = df_ibm['Attrition'].map({'Yes': 1, 'No': 0})
df_ibm['OverTime'] = df_ibm['OverTime'].map({'Yes': 1, 'No': 0})
# Обчислення індексу добробуту
df_ibm['WellBeingLevel'] = df_ibm[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1)

# Підготовка даних
features_ibm = ['WellBeingLevel', 'OverTime', 'MonthlyIncome', 'Age']
df_ibm = df_ibm.dropna(subset=features_ibm + ['Attrition'])
X_clf = df_ibm[features_ibm]
y_clf = df_ibm['Attrition']

# Масштабування
scaler_attr = StandardScaler()
X_clf_scaled = scaler_attr.fit_transform(X_clf)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)

# Класифікаційна модель
classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_clf = {'n_estimators': [100], 'max_depth': [5, 10, None]}
grid_clf = GridSearchCV(classifier, param_grid_clf, cv=5, scoring='f1', n_jobs=-1)
grid_clf.fit(X_train_clf, y_train_clf)
clf_model = grid_clf.best_estimator_

# Збереження важливості ознак
clf_importance = pd.Series(clf_model.feature_importances_, index=features_ibm).sort_values(ascending=False)
clf_importance.to_csv("model/attrition_feature_importance.csv")

plt.figure(figsize=(8,5))
clf_importance.plot(kind='bar', color='orange')
plt.title("Важливість ознак для моделі ризику звільнення")
plt.tight_layout()
plt.savefig("output/classification_feature_importance.png")
plt.close()

# === 5. Оцінка якості моделей ===
# Регресія
y_pred_reg = reg_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# Класифікація
y_pred_clf = clf_model.predict(X_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

# Кластеризація
sil_score = silhouette_score(X_reg_scaled, kmeans.labels_)

# === 6. Збереження моделей ===
dump(scaler, "model/scaler.pkl")
dump(kmeans, "model/cluster_model.pkl")
dump(reg_model, "model/productivity_model.pkl")
dump(clf_model, "model/attrition_model.pkl")
dump(scaler_attr, "model/scaler_attrition.pkl")

# === 7. Вивід результатів ===
print("✅ Усі моделі успішно навчені та збережені!")
print(f"[Регресія] MSE: {mse:.2f}, R²: {r2:.2f}")
print(f"[Класифікація] Accuracy: {acc:.2f}, F1-score: {f1:.2f}")
print(f"[Кластеризація] Silhouette Score: {sil_score:.2f}")

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score

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

# === Оцінка якості регресійної моделі ===
y_pred_reg = regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"[Регресія] MSE: {mse:.2f}, R²: {r2:.2f}")

# === Оцінка класифікаційної моделі ===
y_pred_clf = classifier.predict(X_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
print(f"[Класифікація] Accuracy: {acc:.2f}, F1-score: {f1:.2f}")

# === Оцінка кластеризації ===
silhouette = silhouette_score(X_cluster_scaled, kmeans.labels_)
print(f"[Кластеризація] Silhouette Score: {silhouette:.2f}")


'''

