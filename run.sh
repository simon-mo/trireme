docker run \
    -d \ #daemon mode
    --runtime=nvidia \ # to check gpu stat
    -p 9999:9999 \
    --restart on-failure \
    -v /var/run/docker.sock:/var/run/docker.sock \
    simonmok/scalabel-orch
