image:
	HOST_ADDR=`python -c "import socket;print(socket.getfqdn())"`
	docker build -t simonmok/scalabel-orch -f dockerfiles/OrchestratorDockerfile --build-arg HOST_ADDR=$(HOST_ADDR) .
	docker build -t simonmok/scalabel-mnist -f dockerfiles/MNISTDockerfile  .
deploy:
	bash run.sh
clean:
	docker stop `docker ps --filter label=ai.scalabel.app=orchestrator -q`
