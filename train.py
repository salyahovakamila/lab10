import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def generate_mock_data():
    # Создаем фейковые данные для симуляции датасета, если исходного файла нет под рукой
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'ApplicantIncome': np.random.randint(2000, 10000, n_samples),
        'LoanAmount': np.random.randint(50, 300, n_samples),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
        'Loan_Status': np.random.choice([1, 0], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    # Добавим немного пропусков для этапа очистки
    df.loc[df.sample(frac=0.05).index, 'Credit_History'] = np.nan
    return df


def preprocess_data(df, is_train=True):
    df = df.copy()

    # 1. Очистка пропусков
    if 'Credit_History' in df.columns:
        df['Credit_History'] = df['Credit_History'].fillna(1.0)
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    # 2. Кодирование категориальных признаков
    cat_cols = ['Gender', 'Married', 'Education']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].map({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'Graduate': 1, 'Not Graduate': 0}).fillna(
                0)

    # Отбор признаков (Feature Selection)
    features = ['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']

    if is_train:
        X = df[features]
        y = df['Loan_Status']
        return X, y
    else:
        # Для инференса возвращаем только признаки, сохраняя структуру
        valid_features = [c for c in features if c in df.columns]
        return df[valid_features]


def main():
    # В реальном сценарии: df = pd.read_csv("loan_data.csv")
    df = generate_mock_data()

    X, y = preprocess_data(df, is_train=True)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Алгоритм 1: Логистическая регрессия
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_preds)
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")

    # Алгоритм 2: Случайный лес
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)  # Для дерева масштабирование не обязательно
    rf_preds = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_preds)
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")

    # Выбираем лучшую модель
    if lr_auc > rf_auc:
        print("Выбрана модель: Logistic Regression")
        best_model = lr
        model_payload = {"model": best_model, "scaler": scaler, "model_type": "lr"}
    else:
        print("Выбрана модель: Random Forest")
        best_model = rf
        model_payload = {"model": best_model, "scaler": None, "model_type": "rf"}

    # Сохраняем модель артефактом
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    print("Модель успешно сохранена в best_model.pkl")


if __name__ == "__main__":
    main()