import io
import pickle
import pandas as pd
from typing import List
from fastapi import HTTPException
from minio import Minio
from utils import minio_password, minio_port, minio_user


class MinioHandler:
    """
    Class for handeling minio
    """

    def __init__(self) -> None:
        self.minio_password = minio_password
        self.minio_port = minio_port
        self.minio_user = minio_user
        self.url = f"127.0.0.1:{self.minio_port}"

        # connecting to Minio
        self.MinioClient = Minio(
            endpoint=self.url,
            access_key=self.minio_user,
            secret_key=self.minio_password,
            secure=False,
        )


    def create_bucket(self, bucket_name: str) -> None:
        """
        Creating bucket in minio
        :param bucket_name: bucket name
        :return: None
        """
        if not self.MinioClient.bucket_exists(bucket_name):
            self.MinioClient.make_bucket(bucket_name)
        else:
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} already exsists"
            )

    def get_bucket_obj(self, bucket_name: str) -> List:
        """
        Listing all items per bucket
        :param bucket_name: bucket name
        :return: List of names of all objects per bucket
        """
        if not self.MinioClient.bucket_exists(bucket_name):
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} does not exsist"
            )
        else:
            return self.MinioClient.list_objects(bucket_name)

    def load_model_minio(self, model_name: str, bucket_name: str):
        """
        Loading model from S3
        :param model_name: model name to load
        :param bucket_name: bucket name
        :return: model stored in S3
        """

        if not self.MinioClient.bucket_exists(bucket_name):
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} does not exsist"
            )

        else:
            model = pickle.loads(
                self.MinioClient.get_object(
                    bucket_name=bucket_name, object_name=model_name
                ).read()
            )
            return model

    def load_data(self, data_name: str, bucket_name: str) -> List:
        """
        Loading data from S3
        :param data_name: dataframe name to load
        :param bucket_name: bucket name
        :return: pd.DataFrame of data to be loaded from S3
        """

        if not self.MinioClient.bucket_exists(bucket_name):
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} does not exsist"
            )

        else:
            file = (
                self.MinioClient.get_object(
                    bucket_name=bucket_name, object_name=data_name
                )
                .read()
                .decode("utf-8")
            )
            df = pd.read_csv(file)
            return df.values.tolist()

    def save_data(self, data: List, data_name: str, bucket_name: str) -> None:
        """
        saving data to S3
        :param data: df to be saved
        :param data_name: dataframe name to save
        :param bucket_name: bucket name
        :return: None
        """
        if not self.MinioClient.bucket_exists(bucket_name):
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} does not exsist"
            )
        else:
            csv = data.to_csv(index=False).encode("utf-8")
            self.MinioClient.put_object(
                bucket_name=bucket_name,
                object_name=data_name,
                data=io.BytesIO(csv),
                length=len(csv),
                content_type="application/csv",
            )
            return HTTPException(status_code=200, detail="DF saved")

    def save_model(self, model, model_name: str, bucket_name: str) -> None:
        """
        saving model to S3
        :param model: model to be saved
        :param model_name: model name to save
        :param bucket_name: bucket name
        :return: None
        """
        if not self.MinioClient.bucket_exists(bucket_name):
            raise HTTPException(
                status_code=404, detail=f"Bucket {bucket_name} does not exsist"
            )
        else:
            obj = pickle.dumps(model)
            self.MinioClient.put_object(
                bucket_name=bucket_name,
                object_name=model_name,
                data=io.BytesIO(obj),
                length=len(obj),
            )
        return HTTPException(status_code=200, detail="model saved")
