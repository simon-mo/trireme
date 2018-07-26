from flask import Flask, request
from gpustat import GPUStatCollection
from subprocess import call

app = Flask(__name__)


@app.route("/gpu")
def gpu_stats():
    return GPUStatCollection.new_query().jsonify()


@app.route("/add", methods=["POST"])
def add_model():
    info = request.json()
    cmd = [
        "docker",
        "run",
        "--runtime=nvidia",
        "-d",
        "-p",
        "9998:8765",
        "simonmok/scalabel-mnist",
        "-e",
        f"CUDA_VISIBLE_DEVICES={','.join(info[gpu])}",
    ]
    print(f"Processing {cmd}")
    proc = call(cmd)
    return proc


app.run(host="0.0.0.0", port="9999", threaded=True, debug=True)

