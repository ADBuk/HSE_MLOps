from fastapi import FastAPI
from models_handler import ModelHandler
from typing import List, Dict
import uvicorn
from pydantic import BaseModel, Field
from minio_handler import MinioHandler
import pandas as pd

handler = ModelHandler()
minio_handler = MinioHandler()

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
    handler.model_train(
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


@app.get("/get_everything_from_bucket")
def get_data_from_bucket(bucket_name: str):
    """
    getting bucket info
    :param bucket_name: bucket_name to get info from
    :return: files
    """
    return minio_handler.get_bucket_obj(bucket_name=bucket_name)


@app.post("/get_model_from_bucket")
def get_model_from_bucket(bucket_name: str, model_name: str):
    """
    getting model from S3
    :param bucket_name: bucket_name to get info from
    :param model_name: model name to load
    :return: model
    """
    return minio_handler.load_model_minio(
        bucket_name=bucket_name, model_name=model_name
    )


@app.post("/get_data_from_s3")
def get_df_from_bucket(bucket_name: str, data_name: str) -> List:
    """
    getting model from S3
    :param bucket_name: bucket_name to get info from
    :param data_name: DF  name to load
    :return: pd>DataFrame
    """
    return minio_handler.load_data(bucket_name=bucket_name, data_name=data_name)


@app.get("/create_bucket")
def create_bucket(bucket_name: str) -> None:
    """
    creating S3 bucket
    :param bucket_name: bucket_name to get info from
    :return: None
    """
    minio_handler.create_bucket(bucket_name)


@app.get("/save_model_to_s3")
def save_model_to_s3(bucket_name: str, model, model_name: str) -> None:
    """
    saving model to S3 bucket
    :param bucket_name: bucket_name to save to
    :param model: model to save
    :param model_name: model_name
    :return: None
    """
    minio_handler.save_model(model, model_name, bucket_name)


@app.get("/save_model_to_s3")
def save_data_to_s3(data: List, data_name: str, bucket_name: str) -> None:
    """
    saving data to S3
    :param data: df to be saved
    :param data_name: dataframe name to save
    :param bucket_name: bucket name
    :return: None
    """
    minio_handler.save_data(data, data_name, bucket_name)


# Implement Swagger documentation
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
