import os
import pickle
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

available_models = ["LinReg", "Catboost"]


class ModelHandler:
    """
    class for handling all models
    """

    def __init__(self) -> None:
        self.storage = "mlops_storage"

        if not os.path.exists(self.storage):
            os.makedirs(self.storage)
        self.available_models = available_models

    def load_model(self, model_name: str):
        """
        function for loading pretrained models
        :param model_name: name of the pretrained model
        :return: pretrained model
        """
        model_path = os.path.join(self.storage, f"{model_name}.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def save_model(self, model_name) -> None:
        """
        function for saving trained models
        :param model_name: name of the trained model to save
        :return: None
        """
        model_path = os.path.join(self.storage, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(f)

    def get_all_models(self) -> List:
        """
        getting all the saved models
        :return: List of names of saved models
        """
        return self.available_models

    def model_train(
        self,
        model_name: str,
        x: List,
        y: List,
        params: Dict = None,
        save: bool = False,
    ) -> None:
        """
        def for training certain models with params (optional)
        :param model_name: model_name to train
        :param x: X matrix
        :param y: target
        :param params: params for training
        :param save: Flag whether to save model or not
        :return: None
        """
        if model_name == "LinReg":
            if params is None:
                model = LinearRegression()
                model.fit(x, y)
            else:
                model = LinearRegression(**params)
                model.fit(x, y)
        elif model_name == "Catboost":
            if params is None:
                model = CatBoostRegressor()
                model.fit(x, y)
            else:
                model = CatBoostRegressor(**params)
                model.fit(x, y)
        else:
            raise f"{model_name} is not implemented for training. Available models are: {self.available_models}"
        if save:
            self.save_model(model_name)

    def delete_model(self, model_name: str) -> None:
        """
        delete certain models
        :param model_name: model to delete
        :return: None
        """
        model_path = os.path.join(self.storage, f"{model_name}.pkl")
        os.remove(model_path)

    def get_prediction(self, model_name: str, data: List) -> List:
        """
        get prediction of certain model
        :param model_name: name of model to predict
        :param data: data for predictions
        :return: predictions
        """
        model = self.load_model(model_name)
        predictions = model.predict(data)
        return predictions.to_list()
