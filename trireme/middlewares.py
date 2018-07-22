import redis
import json


class JsonLoadsActor:
    def __init__(self):
        pass

    async def __call__(self, input_batch):
        return [json.loads(inp) for inp in input_batch]


class JsonDumpsActor:
    def __init__(self):
        pass

    async def __call__(self, input_batch):
        return [json.dumps(inp) for inp in input_batch]


class RedisDownloaderActor:
    def __init__(self):
        self.r = redis.Redis()

    async def __call__(self, input_batch):
        """Mutate each input using the rule:
        - If it has a field 'input', ignore
        - If it has a field 'object id', add input field with data from redis
        """
        mutated = []
        for inp in input_batch:
            if "input" in inp:
                mutated.append(inp)
                continue

            oid = inp["object id"]
            redis_key = oid.replace("redis://", "")

            retrieved_input = self.r.get(redis_key)
            if retrieved_input == None:
                raise Exception(redis_key + " not found in redis")
            inp["input"] = retrieved_input
            mutated.append(inp)
        return mutated
