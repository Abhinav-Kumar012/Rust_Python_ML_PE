from locust import HttpUser, task, between
import pandas as pd
import numpy as np

IP = "127.0.0.1"
PORT = "8000"
PATH_TO_DSET = "test_data/ag_news/ag_news_test.parquet"


class LoadTestprofile(HttpUser):
    wait_time = between(0.1, 0.2)
    host = f"http://{IP}:{PORT}"

    # Load dataset once (shared across users)
    df = pd.read_parquet(PATH_TO_DSET)

    @task
    def load_task(self):
        # pick random row
        row = self.df.iloc[np.random.randint(0, len(self.df))]

        # payload matching FastAPI schema
        payload = {
            "text": str(row["text"])
        }

        # send request
        response = self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )

        # minimal error logging
        if response.status_code != 200:
            print("❌ API Error:", response.text)