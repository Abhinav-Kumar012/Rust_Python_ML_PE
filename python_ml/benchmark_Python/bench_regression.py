from locust import HttpUser, task, between
import pandas as pd
import numpy as np

IP = "127.0.0.1"
PORT = "8000"
PATH_TO_DSET = "test_data/regression/cali_housing.parquet"

class LoadTestprofile(HttpUser):
    wait_time = between(0.1, 0.2)
    host = f"http://{IP}:{PORT}"

    def on_start(self):
        # Load once per user (acceptable for now)
        self.df = pd.read_parquet(PATH_TO_DSET)

    @task
    def load_task(self):
        row = self.df.iloc[np.random.randint(0, len(self.df))]

        payload = {
            "MedInc": float(row["MedInc"]),
            "HouseAge": float(row["HouseAge"]),
            "AveRooms": float(row["AveRooms"]),
            "AveBedrms": float(row["AveBedrms"]),
            "Population": float(row["Population"]),
            "AveOccup": float(row["AveOccup"]),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "MedHouseVal": float(row["MedHouseVal"]),
        }

        response = self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )

        # Optional: basic validation
        if response.status_code != 200:
            print("❌ Error:", response.text)