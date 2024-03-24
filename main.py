from pydantic import BaseModel

from fastapi import FastAPI
import joblib
import pandas as pd

class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

file = 'model.joblib'
model = joblib.load(file)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/score")
def score(args: iris):
    data = pd.json_normalize(args.model_dump())
    result = model.predict(data)
    return {"resultat": result[0]}
