import io
import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Mortgage Approval ML Service")

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для хранения загруженной модели в памяти
CURRENT_MODEL = None


class ClientFeatures(BaseModel):
    Gender: int
    Married: int
    Education: int
    ApplicantIncome: float
    LoanAmount: float
    Credit_History: float


class PredictionResponse(BaseModel):
    Gender: int
    Married: int
    Education: int
    ApplicantIncome: float
    LoanAmount: float
    Credit_History: float
    loan_status: int


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global CURRENT_MODEL
    try:
        contents = await file.read()
        model_data = pickle.loads(contents)

        # Минимальная проверка структуры весов
        if not isinstance(model_data, dict) or "model" not in model_data:
            raise HTTPException(status_code=400, detail="Неверный формат структуры pkl файла")

        CURRENT_MODEL = model_data
        return {"status": "успешно", "message": f"Модель {model_data.get('model_type', 'unknown')} успешно загружена"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки модели: {str(e)}")


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(data: List[ClientFeatures]):
    global CURRENT_MODEL
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /upload-model")

    model = CURRENT_MODEL["model"]
    scaler = CURRENT_MODEL["scaler"]

    results = []
    for item in data:
        features_dict = item.model_dump()
        # Превращаем в массив для скалера/модели
        features_arr = [[
            features_dict['Gender'],
            features_dict['Married'],
            features_dict['Education'],
            features_dict['ApplicantIncome'],
            features_dict['LoanAmount'],
            features_dict['Credit_History']
        ]]

        if scaler is not None:
            features_arr = scaler.transform(features_arr)

        prediction = int(model.predict(features_arr)[0])

        response_item = features_dict.copy()
        response_item["loan_status"] = prediction
        results.append(response_item)

    return results


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    global CURRENT_MODEL
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /upload-model")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")

    # Импортируем функцию обработки из train.py для чистоты подхода
    from train import preprocess_data
    from sklearn.metrics import roc_auc_score

    # Проверяем, есть ли в переданном файле истинные таргеты
    has_target = "Loan_Status" in df.columns or "loan_status" in df.columns
    target_col = "Loan_Status" if "Loan_Status" in df.columns else "loan_status"

    # Сохраняем исходный таргет, если он есть
    y_true = df[target_col].copy() if has_target else None

    # Предобработка сырого датасета
    X = preprocess_data(df, is_train=False)

    model = CURRENT_MODEL["model"]
    scaler = CURRENT_MODEL["scaler"]

    # Если модель требует стандартизации
    if scaler is not None:
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

    df["predicted_loan_status"] = preds

    response = {
        "dataset": df.to_dict(orient="records")
    }

    if has_target and y_true is not None:
        try:
            auc = roc_auc_score(y_true, probs)
            response["roc_auc"] = float(auc)
        except:
            response["roc_auc"] = "Не удалось посчитать ROC-AUC (возможно, присутствует только 1 класс)"

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)