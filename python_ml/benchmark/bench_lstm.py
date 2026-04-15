from locust import HttpUser, task, between
import numpy as np

IP = "127.0.0.1"
PORT = "9050"
SEQ_LEN = 4

class LoadTestprofile(HttpUser):
    wait_time = between(0.1,0.2)
    host = f"http://{IP}:{PORT}"
    rng = rng = np.random.default_rng()
    @task
    def load_task(self):
        random_list = self.rng.uniform(0.0, 10.0, SEQ_LEN).tolist()
        _ = self.client.post("/predict",json=random_list)