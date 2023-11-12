from fastapi import FastAPI
from models_handler import ModelHandler
from typing import List, Dict
import uvicorn
from pydantic import BaseModel, Field

handler = ModelHandler()

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})


class TrainingData(BaseModel):
    features: List[List[float]] = Field(
        description="x data for regression",
        example=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    )
    labels: List[float] = Field(
        description="y data for regression",
        example=[1.0, 2.0, 3.0],
    )


@app.get("/")
async def root():
    return "mlops_hw_1"


# training model
@app.post("/train_models")
async def train_model(
    model_name: str,
    data: TrainingData,
    params: Dict = None,
    save: bool = True,
):
    """
    training models with handler
    if model is not in created models and save==True it will automatically create new model then save it
    :param model_name: witch model to train
    :param x: x-data
    :param y: y-data
    :param params: hyperparameters (optional)
    :param save: save model in storage or not
    :return: trained model
    """
    return handler.model_train(
        x=data.features, y=data.labels, model_name=model_name, params=params, save=save
    )


@app.post("/get_prediction")
def predict(model_name: str, data: TrainingData) -> List:
    """
    getting prediction with handler
    :param model_name: which model to use for predictions
    :param data: data to make predictions
    :return: preds
    """
    return handler.get_prediction(model_name=model_name, data=data.features)


@app.get("/get_all_models")
def get_all_models() -> List:
    """
    getting all existing models
    :return: list of all trained models
    """
    return handler.get_all_models()


@app.post("/delete_model/")
def delete_model(model_name: str):
    """
    deleting model with handler
    :param model_name: which model to delete
    :return: None
    """
    handler.delete_model(model_name=model_name)


# Implement Swagger documentation
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
