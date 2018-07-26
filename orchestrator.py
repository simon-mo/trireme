from flask import Flask, request, jsonify
from gpustat import GPUStatCollection
from subprocess import call
import sqlite3
import docker
import jsonschema
from typing import Tuple, Union
import socket
from orch_sql import get_image_from_model

# Global Objects
app = Flask(__name__)
conn = sqlite3.connect("models.db")
docker_client_high_level = docker.from_env()
docker_client_low_level = docker.APIClient(base_url="unix://var/run/docker.sock")


# JSON Schema

# Schema for Registering/Adding Model
# Example:
# {
#  "modelName" : "resnet18",
#  "gpu": [0,3,4]
# }
add_model_schema = {
    "type": "object",
    "properties": {
        "modelName": {"type": "string"},
        "gpu": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["modelName", "gpu"],
}


def _validate_add_model_json(input_json: dict) -> Tuple[bool, Union[None, str]]:
    try:
        jsonschema.validate(input_json, add_model_schema)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, e.message


@app.route("/gpustat", methods=["GET"])
def gpu_stats():
    d = GPUStatCollection.new_query().jsonify()
    return jsonify(d)


# From https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python/2838309#2838309
def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))  # Bind to a free port provided by the host.
    port_num = s.getsockname()[1]  # Return the port number assigned.
    s.close()
    return port_num


def _find_free_port_above_10000() -> int:
    port_num = _find_free_port()
    while port_num < 10000:
        port_num = _find_free_port()
    return port_num


def _get_host() -> str:
    return socket.getfqdn()


# Return
# {
#   'success': True,
#   'wsAddr': 'ws://ip:port'
# } OR
# {
#   'success': False,
#   'reason': "Json invalid, missing field name..."
# }
@app.route("/add_model", methods=["POST"])
def add_model():
    info = request.json

    # Check for json schema
    is_valid, msg = _validate_add_model_json(info)
    if not is_valid:
        return jsonify({"success": False, "reason": msg})

    # Check for docker image
    # NOTE(simon) July 26, 2018:
    #    This just performs a lookup against sqlite db.
    #    c.f. notes in orch_sql.py, this is meant to talk to remote db in
    #    the next few iteration.
    model_name = info["modelName"]
    image_name = get_image_from_model(model_name)
    if image_name is None:
        return jsonify(
            {
                "success": False,
                "reason": f"Docker Image for model {model_name} not found in database!",
            }
        )

    cuda_str = ",".join([str(i) for i in info["gpu"]])

    host = _get_host()

    running_models = docker_client_high_level.container.list(
        filters={"label": f"ai.scalabel.model={model_name}"}
    )
    if len(running_models) == 0:
        ws_port = _find_free_port_above_10000()
        docker_client_high_level.containers.run(
            image=image_name,
            runtime="nvidia",
            environment={"CUDA_VISIBLE_DEVICES": cuda_str},
            ports={"8765": ws_port},
            labels={"ai.scalabel.model": model_name},
        )
    else:
        container = running_models[0]
        port_info = docker_client_low_level.inspect_container(container.id)[
            "NetworkSettings"
        ]["Ports"]
        # port_info will look like:
        # {'9999/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '9999'}]}
        port = list(port_info.values())[0][0]["HostPort"]
        ws_port = int(port)
    ws_url = f"ws://{host}:{ws_port}"

    return {"success": True, "wsAddr": ws_url}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9999", threaded=True, debug=True)

