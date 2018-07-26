import asyncio
import logging

from trireme.examples.mnist import MnistTrainActor
from trireme.middlewares import ImageDownloaderActor, JsonDumpsActor, JsonLoadsActor
from trireme.pipe import Pipe
from trireme.server import get_server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


loop = asyncio.get_event_loop()
server, req_queue, resp_queue = get_server()

loads = Pipe.new(JsonLoadsActor)
download = Pipe.new(ImageDownloaderActor)
mnist = Pipe.new(MnistTrainActor)
dumps = Pipe.new(JsonDumpsActor)

Pipe(req_queue) > loads > download > mnist > dumps > Pipe(resp_queue)

loads = loads.get()
download = download.get()
mnist = mnist.get()
dumps = dumps.get()

loop.run_until_complete(asyncio.gather(server, download(), mnist(), loads(), dumps()))
loop.run_forever()
