from locust import HttpUser, task, between,events
import pandas as pd
import numpy as np

IP = "127.0.0.1"
PORT = "9050"
PATH_TO_DSET = "test_data/ag_news/ag_news_test.parquet"

'''
# download outside the script

import pandas as pd

df = pd.read_parquet(
    "hf://datasets/fancyzhx/ag_news/data/test-00000-of-00001.parquet"
)
df.to_parquet("ag_news_test.parquet")
'''

class LoadTestprofile(HttpUser):
    wait_time = between(0.1,0.2)
    host = f"http://{IP}:{PORT}"

    @task
    def load_task(self):
        random_row = self.df.iloc[np.random.randint(0,len(self.df))]
        text_to_send = {
            'text' : random_row['text']
        }
        response = self.client.post("/predict",json=text_to_send)
        is_correct = (response.json()['prediction'] == self.dict_class[random_row['label']])
        events.request.fire(
            request_type="ML",
            name="accuracy",
            response_time=0,
            response_length=1,
            exception=None if is_correct else Exception("wrong"),
        )
        

    def on_start(self):
        self.df = pd.read_parquet(PATH_TO_DSET)
        self.dict_class = {
            0 : "World",
            1 : "Sports",
            2 : "Business",
            3 : "Technology"
        }