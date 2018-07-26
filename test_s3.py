"""
This demo work as follows:
- infer_request_producer will start sending a inference request every 0-2 seconds
- infer_response_consumer will listen to the websocket and receive prediction. If the prediction 
  matches our label, it will put a cancellation request in the queue; else, it will put a training
  request with the correct label in the queue.
- last_request_handler will consume the third request queue
"""

import asyncio
import websockets
import torch
from torchvision import transforms, datasets
import json
import numpy as np
import redis
import hashlib
import sys
import time

feedback_queue = asyncio.Queue(maxsize=10000)
oid_to_label = {}


oid_start_time = {}

SLEEP_TIME = 1
if len(sys.argv) > 1:
    SLEEP_TIME = float(sys.argv[1])

import requests
import numpy as np


class MnistGenInputActor:
    def __init__(self):
        raw = requests.get("https://s3.amazonaws.com/simon-mo-dev-public/index.txt").text
        lst = raw.split("\n")
        self.urls = [
            item.strip()
            for item in lst
            if item.startswith("https") and item.endswith(".png")
        ]

    def __call__(self):
        url = np.random.choice(self.urls)
        print("sending ", url)
        wrapped = {"object id": url, "label": int(np.random.randint(0, 9))}

        return wrapped


async def infer_request_producer(websocket):
    generator = MnistGenInputActor()

    while True:
        inp = generator()
        pred_req = {"path": "infer", "object id": inp["object id"]}
        serialized = json.dumps(pred_req)

        oid_start_time[inp["object id"]] = time.time()
        await websocket.send(serialized)
        oid_to_label[inp["object id"]] = inp["label"]

        await asyncio.sleep(SLEEP_TIME)


async def infer_response_consumer(websocket):
    while True:
        resp = await websocket.recv()
        resp_dict = json.loads(resp)
        inp_oid = resp_dict["object id"]

        print(
            f"Prediction {inp_oid} took {time.time() - oid_start_time[inp_oid]} seconds"
        )

        pred = int(resp_dict["prediction"])
        label = int(oid_to_label[inp_oid])

        if pred == label:
            await feedback_queue.put(
                {
                    "path": "cancel",
                    "model state id": resp_dict["model state id"],
                    "object id": inp_oid,
                }
            )
        else:
            await feedback_queue.put(
                {
                    "path": "train",
                    "model state id": resp_dict["model state id"],
                    "label": label,
                    "object id": inp_oid,
                }
            )


async def feedback_queue_consumer(websocket):
    while True:
        next_req = await feedback_queue.get()

        serialized = json.dumps(next_req)
        await websocket.send(serialized)


async def main():
    async with websockets.connect("ws://localhost:8765/") as websocket:
        asyncio.ensure_future(feedback_queue_consumer(websocket))
        asyncio.ensure_future(infer_response_consumer(websocket))
        asyncio.ensure_future(infer_request_producer(websocket))

        while True:
            await asyncio.sleep(100)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
