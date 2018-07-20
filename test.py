import asyncio
import websockets

async def hello(times):
    async with websockets.connect(
            'ws://localhost:8765') as websocket:
        for i in range(times):
            await websocket.send(f"This is input {i}")
            print(f"> {i}")
        for i in range(times):
            resp = await websocket.recv()
            print(f"< {resp}")

asyncio.get_event_loop().run_until_complete(hello(20))