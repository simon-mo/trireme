from flask import Flask, request, jsonify
from gpustat import GPUStatCollection
from subprocess import call

app = Flask(__name__)


@app.route("/gpustat")
def gpu_stats():
    d = GPUStatCollection.new_query().jsonify()
    return jsonify(d)


@app.route("/add", methods=["POST"])
def add_model():
    info = request.json
    cmd = [
        "docker",
        "run",
        "--runtime=nvidia",
        "-d",
        "-p",
        "9998:8765",
        "-e",
        "CUDA_VISIBLE_DEVICES={}".format(','.join(info['gpu'])),
        "simonmok/scalabel-mnist",
    ]
    print("Processing", "cmd")
    proc = call(cmd)
    return 'ws://localhost:9998'


app.run(host="0.0.0.0", port="9999", threaded=True, debug=True)

