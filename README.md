# AI-аналітика продуктивності та добробуту працівників
Цей проєкт - дипломна робота студентки 6го курсу спеціальності "Інженерія програмного забезпечення" Пархоменко Юлії, присвячена створенню системи аналізу добробуту та прогнозування ефективності працівників з використанням алгоритмів машинного навчання.

# Мета
- Передбачити **продуктивність працівника** за допомогою регресійної моделі.
- Виявити **ризик звільнення** (класифікація).
- Провести **кластеризацію** працівників для виявлення поведінкових груп.
- Побудувати інтерактивні **візуалізації** та графіки для аналізу.

# Дані
У проєкті використано два набори даних із Kaggle
1. **IBM HR Employee Analytics Attrition & Performance Dataset**  
   https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
   Дані про співробітників (позиція, досвід, задоволення роботою, рівень доходу тощо) - використано для класифікації ризику звільнення.
2. **Remote Work Productivity Dataset**  
   https://www.kaggle.com/datasets/mrsimple07/remote-work-productivity
   Дані опитування про продуктивність працівників на віддаленій роботі (добробут, перевтома, умови праці) - використано для регресійного аналізу та кластеризації.

Дані були попередньо очищені, масштабовані та оброблені перед тренуванням моделей.


# Використані технології
- Python (scikit-learn, pandas, matplotlib, seaborn)
- PCA, кластеризація, Random Forest, Logistic Regression
- Flask (у `app.py`) - базовий інтерфейс для демонстрації моделей
- Збереження моделей у `.pkl`

# Структура проєкту
- `train_models.py` - навчання моделей (класифікація, регресія, кластеризація)
- `app.py` - запуск Flask-сервера з інтерфейсом
- `visualization.py` - побудова графіків
- `*.pkl` - збережені моделі та скейлери
- `visualization_summary.png` - підсумкові візуалізації

# Приклад результатів
- Кореляція між показниками добробуту та ефективністю
- Класифікація ризику звільнення з точністю > X%
- Регресійне передбачення продуктивності
- Виявлення кластерів працівників

# Як запустити
```bash
# 1. Створити віртуальне середовище
python -m venv venv
# 2. Активувати середовище
.\venv\Scripts\activate       # для Windows
# source venv/bin/activate    # для MacOS/Linux
# 3. Встановити залежності
pip install -r requirements.txt
# 4. Навчити моделі
python train_models.py
# 5. Запустити застосунок
python app.py
