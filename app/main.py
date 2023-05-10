from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

model = joblib.load("lgbm_model.joblib")

type_mapping = {"CASHOUT": 0, "DEBIT": 0, "PAYMENT": 0, "CASHIN": 1, "TRANSFER": 1}

class InputFeatures(BaseModel):
    step: int = Field(..., description="The hour at which the transaction occurred.")
    type: str = Field(..., description="The type of transaction (e.g., Cash out=0, Transfer=1).")
    amount: float = Field(..., description="The amount involved in the transaction.")
    oldbalanceOrig: float = Field(...,description="The balance before the transaction from the origin account.")
    newbalanceOrig: float = Field(..., description="The balance after the transaction from the origin account.")
    oldbalanceDest: float = Field(..., description="The balance before the transaction in the destination account.")
    newbalanceDest: float = Field(..., description="The balance after the transaction in the destination account.")
    errorBalanceDest: float = Field(..., description="The difference between the newbalanceDest and the expected balance.")
    errorBalanceOrig: float = Field(..., description="The difference between the newbalanceOrig and the expected balance.")

class PredictionResult(BaseModel):
    fraud_probability: float
    prediction: str

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict", response_model=dict)
def predict_transaction(transaction: InputFeatures):
    transaction_type = transaction.type.upper()

    try:
        transaction_dict = transaction.dict()
        transaction_dict['type'] = type_mapping[transaction_type]  
    except KeyError:
        return {"error": f"Invalid value '{transaction_type}' for 'type'. Please choose from 'CASHOUT', 'DEBIT', 'PAYMENT', 'CASHIN', or 'TRANSFER'."}

    transaction_array = np.array(list(transaction_dict.values())).reshape(1, -1)

    # Perform prediction
    prediction_probabilities = model.predict_proba(transaction_array)[0]
    fraud_probability = prediction_probabilities[1]
    genuine_probability = prediction_probabilities[0]
    prediction = "fraud" if fraud_probability > genuine_probability else "genuine"
    prediction_result = PredictionResult(fraud_probability=fraud_probability, prediction=prediction)

    response_statement = f"The probability of the transaction being genuine is: {genuine_probability*100:.2f}%. "
    response_statement += f"The probability of the transaction being fraudulent is: {fraud_probability*100:.2f}%. "
    response_statement += f"The transaction can be classified as a: {prediction.capitalize()} transaction."

    return {"prediction_result": prediction_result, "response_statement": response_statement}