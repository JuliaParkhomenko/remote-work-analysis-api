import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Завантаження даних
df_remote = pd.read_csv("remote-work-productivity.csv", delimiter=";")
df_remote['EmploymentType'] = df_remote['EmploymentType'].map({'Remote': 0, 'In-Office': 1})

# Генерація ознак
def weighted_scale(wellbeing, productivity):
    combined = (0.6 * wellbeing + 0.4 * productivity) / 100
    score = round(combined * 3 + 1 + np.random.normal(0, 0.3))
    return int(np.clip(score, 1, 4))

df_remote['JobSatisfaction'] = df_remote.apply(lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)
df_remote['EnvironmentSatisfaction'] = df_remote.apply(lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)
df_remote['WorkLifeBalance'] = df_remote.apply(lambda row: weighted_scale(row['WellBeingScore'], row['ProductivityScore']), axis=1)

# Масштабування
features = ['HoursWorkedPerWeek', 'WellBeingScore', 'JobSatisfaction',
            'EnvironmentSatisfaction', 'WorkLifeBalance', 'EmploymentType']
X = df_remote[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Завантаження важливостей
reg_feat = pd.read_csv("model/productivity_feature_importance.csv", index_col=0)
clf_feat = pd.read_csv("model/feature_importance.csv", index_col=0)

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Feature Importance (Regression) - Для продуктивності
reg_feat.sort_values(by=reg_feat.columns[0]).plot(kind='barh', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title("Важливість ознак для прогнозування продуктивності (Регресія)")
axes[0, 0].set_xlabel("Значущість")

# Feature Importance (Classification) - Для ризику звільнення
clf_feat.sort_values(by=clf_feat.columns[0]).plot(kind='barh', ax=axes[0, 1], color='orange')
axes[0, 1].set_title("Важливість ознак для класифікації ризику звільнення (Класифікація)")
axes[0, 1].set_xlabel("Значущість")

# PCA-кластери
axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)
axes[1, 0].set_title("Кластери працівників (PCA)")
axes[1, 0].set_xlabel("PCA 1")
axes[1, 0].set_ylabel("PCA 2")

# Productivity ~ WellBeing
sns.regplot(data=df_remote, x='WellBeingScore', y='ProductivityScore', ax=axes[1, 1])
axes[1, 1].set_title("Залежність Продуктивності від Добробуту")

plt.tight_layout()
plt.savefig("visualization_summary.png")
plt.show()


















'''
# Для продуктивності
# Завантаження важливості ознак регресійної моделі
reg_feat = pd.read_csv("model/productivity_feature_importance.csv", index_col=0)
reg_feat = reg_feat.sort_values(by=reg_feat.columns[0])

# Побудова графіка
reg_feat.plot(kind='barh', legend=False, title='Важливість ознак для прогнозування продуктивності')
plt.xlabel("Значущість")
plt.tight_layout()
plt.show()

# Для ризику звільнення
# Завантаження важливості ознак класифікаційної моделі
clf_feat = pd.read_csv("model/feature_importance.csv", index_col=0)
clf_feat = clf_feat.sort_values(by=clf_feat.columns[0])  # Вказуємо стовпець за яким сортувати

# Побудова графіка
clf_feat.plot(kind='barh', legend=False, title='Важливість ознак для класифікації ризику звільнення', color='orange')
plt.xlabel("Значущість")
plt.tight_layout()
plt.show()

'''