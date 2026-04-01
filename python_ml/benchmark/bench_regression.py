from locust import HttpUser, task, between
import pandas as pd
import numpy as np

IP = "127.0.0.1"
PORT = "9050"
PATH_TO_DSET = "test_data/regression/cali_housing.parquet"

'''
# download outside the script

import pandas as pd

df = pd.read_parquet(
    "hf://datasets/gvlassis/california_housing/data/test-00000-of-00001.parquet"
)
df.to_parquet("cali_housing.parquet")
'''

class LoadTestprofile(HttpUser):
    wait_time = between(0.1,0.2)
    host = f"http://{IP}:{PORT}"

    @task
    def load_task(self):
        random_row = self.df.iloc[np.random.randint(0,len(self.df))]
        text_to_send = {
            'MedInc' : random_row['MedInc'],
            'HouseAge' : random_row['HouseAge'],
            'AveRooms' : random_row['AveRooms'],
            'AveBedrms' : random_row['AveBedrms'],
            'Population' : random_row['Population'],
            'AveOccup' : random_row['AveOccup'],
            'Latitude' : random_row['Latitude'],
            'Longitude' : random_row['Longitude'],
            'MedHouseVal' : random_row['MedHouseVal']
        }
        _ = self.client.post("/predict",json=text_to_send)
        # is_correct = (response.json()['prediction'] == self.dict_class[random_row['label']])
        
        

    def on_start(self):
        self.df = pd.read_parquet(PATH_TO_DSET)