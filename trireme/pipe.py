import asyncio
from functools import partial

from .server import QUEUE_MAX_SIZE


def sanitize_batch(batch):
    sanitized = []
    for m in batch:
        if not isinstance(m, dict):
            sanitized.append(m)
            continue
        s = m.copy()
        if "input" in s and s["input"] is not None:
            s["input"] = f"length {len(s['input'])} bytes"
        sanitized.append(s)
    return sanitized


async def pipe_factory(req_queue, resp_queue, actor_cls, args, kwargs):
    actor = actor_cls(*args, **kwargs)
    while True:
        if req_queue.empty():
            await asyncio.sleep(0.1)
            continue

        msg_batch = []
        for _ in range(req_queue.qsize()):
            msg = await req_queue.get()
            msg_batch.append(msg)

        results = await actor(msg_batch)

        for result in results:
            await resp_queue.put(result)


class Pipe(object):
    @classmethod
    def new(cls, func, *args, **kwargs):
        return Pipe(partial(pipe_factory, actor_cls=func, args=args, kwargs=kwargs))

    def __init__(self, func_or_queue):
        self.is_func = False
        self.is_queue = False
        self.queue = None
        self.func = None

        if isinstance(func_or_queue, asyncio.Queue):
            self.is_queue = True
            self.queue = func_or_queue
        elif callable(func_or_queue):
            self.is_func = True
            self.func = func_or_queue
            self.wrapped_func = func_or_queue
        else:
            raise Exception("Input is neither a callable nor a queue.")

    def __gt__(self, other):  # >
        assert isinstance(other, self.__class__), "Chaining Object must be Pipe Object."

        if self.is_queue and other.is_func:
            self.handle_queue_to_func(other)
        elif self.is_func and other.is_func:
            self.handle_func_to_func(other)
        elif self.is_func and other.is_queue:
            self.handle_func_to_queue(other)
        else:
            raise NotImplementedError("Do not support queue to queue chaining")

        return other

    def handle_queue_to_func(self, func):
        func.wrap_func(req_queue=self.queue)

    def handle_func_to_func(self, func):
        queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self.wrap_func(resp_queue=queue)
        func.wrap_func(req_queue=queue)

    def handle_func_to_queue(self, queue):
        self.wrap_func(resp_queue=queue.get())

    def wrap_func(self, req_queue=None, resp_queue=None):
        assert self.is_func, "Must be a func"
        wrapped = self.wrapped_func

        if req_queue:
            wrapped = partial(wrapped, req_queue=req_queue)
        if resp_queue:
            wrapped = partial(wrapped, resp_queue=resp_queue)

        self.wrapped_func = wrapped

    def get(self):
        if self.is_func:
            return self.wrapped_func
        else:
            return self.queue
