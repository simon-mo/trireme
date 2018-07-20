import asyncio
from server import get_server
from pipe import Pipe, pipe_factory

loop = asyncio.get_event_loop()
server, req_queue, resp_queue = get_server()


async def hi(inp_batch, tag):
    print(tag, f"len {len(inp_batch)}")
    return inp_batch


one = Pipe.new(hi, tag="one")
two = Pipe.new(hi, tag="two")

Pipe(req_queue) > one > two > Pipe(resp_queue)

one = one.get()
two = two.get()

loop.run_until_complete(asyncio.gather(server, one(), two()))
loop.run_forever()
