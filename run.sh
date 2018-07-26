docker run -d --runtime=nvidia -p 9999:9999 --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock --label ai.scalabel.app=orchestrator simonmok/scalabel-orch
