from locust import HttpUser, task, between
from pathlib import Path
from random import randint

IMAGE_PATH = "test_data/mnist"
IP = "127.0.0.1"
PORT = "8000"  # ✅ match FastAPI

class LoadTestprofile(HttpUser):
    wait_time = between(0.1, 0.2)
    host = f"http://{IP}:{PORT}"

    def on_start(self):
        self.test_image_paths = [
            p for p in Path(IMAGE_PATH).iterdir() if p.is_file()
        ]

    @task
    def load_task(self):
        img_path = self.test_image_paths[randint(0, len(self.test_image_paths) - 1)]

        with open(img_path, "rb") as img:
            files = {
                "file": (  
                    img_path.name,
                    img,
                    "image/jpeg"
                )
            }

            self.client.post(
                "/predict",
                files=files,
                name="/predict"
            )