# Trireme

![Trimeme](https://cdn.britannica.com/668x448/70/185470-004-DE86BA21.jpg)

- A websocket based training/inference server.
- Flexible asyncio based pipeline execution framework.

## Example
0. Make sure you have `redis-server` running at port 6379. If not, you can do `docker run -p 6379:6379 -d redis`. (Warning: `test.py` run a flushall, it will erase your redis db!!!)
1. Setup environment by running `pipenv install --dev`
2. Enter the virtualenv by running `pipenv shell`
3. Inside of virtualenv, run:
    - `python app.py` will start the trireme server. 
    - `python test.py [interval]` will start sending requests. You can specify shorter interval in seconds for simulate real work load. 


## API
Checkout `API.md`
