from main import app
from fastapi.testclient import TestClient

test_client = TestClient(app)

def test_train_model():
    hyperparameters = None
    training_data = {
        "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0]],
        "labels": [2.5, 3.5, 4.5, 5.5],
    }
    response = test_client.post(
        "http://0.0.0.0:8000/train_models",
        json={
            "hyperparameters": hyperparameters,
            "training_data": training_data,
            "save": True,
            "model_name": "LinReg",
        },
    )
    print(response.status_code)
    data = response.json()

    return data


print(test_train_model())


def test_get_model():
    response = test_client.get("http://0.0.0.0:8000/get_all_models")
    print(response.status_code)
    data = response.json()

    return data


print(test_get_model())


def delete_model(model_name="LinReg"):
    response = test_client.post(
        f"/delete_model",
        json={
            "model_name": f"{model_name}.pkl",
        },
    )

    print(response.status_code)
    data = response.json()

    return data


print(delete_model())
