image:
	docker build -t simonmok/scalabel-orch -f dockerfiles/OrchestratorDockerfile  .
	docker build -t simonmok/scalabel-mnist -f dockerfiles/MNISTDockerfile  .
deploy:
	bash run.sh
clean:
	docker stop `docker ps --filter label=ai.scalabel.app=orchestrator -q`
