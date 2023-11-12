from fastapi.testclient import TestClient
from flask_api import app

client = TestClient(app)


def test_train_model():
    hyperparameters = {"fit_intercept": True, "copy_X": True}
    training_data = {
        "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "labels": [2.5, 3.5, 4.5],
    }
    response = client.post(
        "http://localhost:8000/train_models",
        json={
            "hyperparameters": hyperparameters,
            "training_data": training_data,
            "save": True,
            "model_name": "LinReg",
        },
    )
    print(response.status_code)
    assert response.status_code == 200
    data = response.json()


test_train_model()
