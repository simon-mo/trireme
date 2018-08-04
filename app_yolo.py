import asyncio
from trireme.server import get_server
from trireme.pipe import Pipe, pipe_factory
from trireme.examples.yolov3 import YoloV3TrainActor
from trireme.middlewares import ImageDownloaderActor, JsonDumpsActor, JsonLoadsActor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


loop = asyncio.get_event_loop()
server, req_queue, resp_queue = get_server()

loads = Pipe.new(JsonLoadsActor)
# redis = Pipe.new(RedisDownloaderActor)
download = Pipe.new(ImageDownloaderActor)
yolo = Pipe.new(YoloV3TrainActor)
dumps = Pipe.new(JsonDumpsActor)

Pipe(req_queue) > loads > download > yolo > dumps > Pipe(resp_queue)

loads = loads.get()
download = download.get()
yolo = yolo.get()
dumps = dumps.get()

loop.run_until_complete(asyncio.gather(server, download(), yolo(), loads(), dumps()))
loop.run_forever()
