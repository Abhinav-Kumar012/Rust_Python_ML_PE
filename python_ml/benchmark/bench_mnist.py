import time
from locust import HttpUser, task, between
from pathlib import Path
from random import randint
IMAGE_PATH = "test_data/mnist"
IP = "127.0.0.1"
PORT = "9050"


class LoadTestprofile(HttpUser):
    wait_time = between(0.1,0.2)
    host = f"http://{IP}:{PORT}"
    test_image_paths = []

    @task
    def load_task(self):
        index = randint(0,len(self.test_image_paths)-1)
        img_path = self.test_image_paths[index]
        with open(self.test_image_paths[index],'rb') as img :
            file = {"image": (
                        img_path.name,
                        img,
                        "image/jpeg"
                    )}
            # start_time = time.time_ns()
            self.client.post("/predict",files=file)
            # total_time = (time.time_ns() - start_time)/1000
        

    def on_start(self):
        for path_img in Path(IMAGE_PATH).iterdir() :
            if path_img.is_file():
                self.test_image_paths.append(path_img.absolute())
        # print(self.test_image_paths)