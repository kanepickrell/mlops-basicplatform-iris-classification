from fastapi import FastAPI
import uvicorn
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()


mlflow.set_tracking_uri("http://www.kanepickrel.com/mlflow")
# Load the model from the remote server
model = mlflow.pyfunc.load_model("runs:/2ef708a8a877475eaf1c32ec91d685e7/model")

class FlowerPartSize(BaseModel):
    width: float
    length: float

class PredictRequest(BaseModel):
    sepal: FlowerPartSize
    petal: FlowerPartSize

@app.post("/predict")
def predict(request: PredictRequest):
    X= pd.DataFrame(
        columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
        data = [[request.sepal.length, request.sepal.width, request.petal.length, request.petal.width]]
    )

    y_proba = model.predict(X)

    flower = int(np.argmax(y_proba))
    return {"flower": flower}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()