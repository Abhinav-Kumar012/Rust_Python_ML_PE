from locust import HttpUser, task, between
import numpy as np

IP = "127.0.0.1"
PORT = "8000"
SEQ_LEN = 4

class LoadTestprofile(HttpUser):
    wait_time = between(0.1, 0.2)
    host = f"http://{IP}:{PORT}"

    def on_start(self):
        # initialize RNG once per user
        self.rng = np.random.default_rng()

    @task
    def load_task(self):
        # generate list[float]
        random_list = self.rng.uniform(0.0, 10.0, SEQ_LEN).tolist()

        # send EXACT format FastAPI expects
        self.client.post(
            "/predict",
            json=random_list,  # ✅ correct (raw list)
            headers={"Content-Type": "application/json"},
            name="/predict"
        )