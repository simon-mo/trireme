import asyncio
from trireme.server import get_server
from trireme.pipe import Pipe, pipe_factory
from trireme.examples.yolov3 import YoloV3TrainActor
from trireme.middlewares import RedisDownloaderActor, JsonDumpsActor, JsonLoadsActor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


loop = asyncio.get_event_loop()
server, req_queue, resp_queue = get_server()

loads = Pipe.new(JsonLoadsActor)
redis = Pipe.new(RedisDownloaderActor)
yolo = Pipe.new(YoloV3TrainActor)
dumps = Pipe.new(JsonDumpsActor)

Pipe(req_queue) > loads > redis > yolo > dumps > Pipe(resp_queue)

loads = loads.get()
redis = redis.get()
yolo = yolo.get()
dumps = dumps.get()

loop.run_until_complete(asyncio.gather(server, redis(), yolo(), loads(), dumps()))
loop.run_forever()
