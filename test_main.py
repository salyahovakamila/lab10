import pytest
import pickle
import io
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def mock_model_bytes():
    # Создаем минимальную заглушку обученной модели для тестов API
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array([[1, 1, 1, 5000, 150, 1.0], [0, 0, 0, 2000, 300, 0.0]])
    y = np.array([1, 0])
    model = LogisticRegression().fit(X, y)

    payload = {"model": model, "scaler": None, "model_type": "lr"}
    return pickle.dumps(payload)


def test_predict_without_model_error():
    # Метод predict должен возвращать ошибку 400, если модель ещё не загружена
    response = client.post("/predict", json=[{
        "Gender": 1, "Married": 1, "Education": 1,
        "ApplicantIncome": 4000, "LoanAmount": 100, "Credit_History": 1.0
    }])
    assert response.status_code == 400
    assert "Модель не загружена" in response.json()["detail"]


def test_upload_and_predict_flow(mock_model_bytes):
    # Тест загрузки модели
    file_payload = {"file": ("best_model.pkl", mock_model_bytes, "application/octet-stream")}
    upload_resp = client.post("/upload-model", files=file_payload)
    assert upload_resp.status_code == 200
    assert upload_resp.json()["status"] == "успешно"

    # Тест предсказания по одному клиенту
    predict_payload = [{
        "Gender": 1, "Married": 1, "Education": 1,
        "ApplicantIncome": 6000, "LoanAmount": 120, "Credit_History": 1.0
    }]
    pred_resp = client.post("/predict", json=predict_payload)
    assert pred_resp.status_code == 200
    assert "loan_status" in pred_resp.json()[0]


def test_predict_from_csv(mock_model_bytes):
    # Формируем фейковый CSV
    csv_data = "Gender,Married,Education,ApplicantIncome,LoanAmount,Credit_History,Loan_Status\nMale,Yes,Graduate,5000,150,1.0,1\nFemale,No,Not Graduate,2000,90,0.0,0"
    file_payload = {"file": ("test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}

    response = client.post("/predict-from-csv", files=file_payload)
    assert response.status_code == 200
    assert "dataset" in response.json()
    assert "roc_auc" in response.json()