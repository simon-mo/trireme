"""
Provide get_server method to establish a basic connection
"""


import asyncio
import websockets
from functools import partial

QUEUE_MAX_SIZE = 10000

import logging

logger = logging.getLogger(__name__)


async def consumer_handler_factory(websocket, path, req_queue):
    while True:
        message = await websocket.recv()
        if message == 'ping':
            await websocket.pong()
        # logging.info(f"\nReceived Input {message}")
        await req_queue.put(message)


async def producer_handler_factory(websocket, path, resp_queue):
    while True:
        message = await resp_queue.get()
        # logging.info(f"Sending Output {message}\n")
        await websocket.send(message)


async def connection_handler_factory(
    websocket, path, consumer_handler, producer_handler
):
    consumer_task = asyncio.ensure_future(consumer_handler(websocket, path))

    producer_task = asyncio.ensure_future(producer_handler(websocket, path))

    done, pending = await asyncio.wait(
        [consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()


def get_server(
    ip="localhost",
    port=8765,
    req_queue=None,
    resq_queue=None,
    queue_size=QUEUE_MAX_SIZE,
):
    if not req_queue:
        req_queue = asyncio.Queue(maxsize=queue_size)
    if not resq_queue:
        resp_queue = asyncio.Queue(maxsize=queue_size)

    consumer_handler = partial(consumer_handler_factory, req_queue=req_queue)
    producer_handler = partial(producer_handler_factory, resp_queue=resp_queue)
    connection_handler = partial(
        connection_handler_factory,
        consumer_handler=consumer_handler,
        producer_handler=producer_handler,
    )

    start_server_co = websockets.serve(connection_handler, ip, port)

    return start_server_co, req_queue, resp_queue
