#!/bin/bash

# if node rank is 0, start ray as head
if [ $NODE_RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT --dashboard-host=$RAY_MASTER_ADDR --disable-usage-stats
    sleep 10
else
    # wait for ray head to start
    sleep 10
    ray start --address=$RAY_MASTER_ADDR:$MASTER_PORT --disable-usage-stats

    # graceful automatic ray client exit
    while true; do
        sleep 300 # check every 5 mins

        status_code=$(ray status > /dev/null 2>&1; echo $?)

        # ray cluster down
        if [ $status_code -ne 0 ]; then
            echo "Ray cluster is down. Exiting..."
            exit 0 # do not remove this line
        fi
    done
fi

# check if ray is running on all nodes
ray status