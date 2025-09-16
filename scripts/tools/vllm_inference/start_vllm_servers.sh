#!/bin/bash

# Set defaults
DTYPE=${DTYPE:-bfloat16}
API_KEY=${API_KEY:-""}
SEED=${SEED:-42}
NODES=${NODES:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
VLLM_LOG_FILE=${VLLM_LOG_FILE:-/dev/null}
API_SERVER_COUNT=${API_SERVER_COUNT:-1}

# VLLM Health monitoring defaults
SERVER_HEARTBEAT_INTERVAL=${SERVER_HEARTBEAT_INTERVAL:-15}       # seconds between periodic health checks after up
SERVER_MAX_WAIT_START=${SERVER_MAX_WAIT_START:-3600}             # max seconds waiting for initial healthy state
SERVER_RESTART_ON_FAILURE=${SERVER_RESTART_ON_FAILURE:-1}        # 1 to enable automatic restart logic
SERVER_MAX_RESTARTS=${SERVER_MAX_RESTARTS:-5}                    # maximum restart attempts before giving up
SERVER_RESTART_WAIT_TIMEOUT=${SERVER_RESTART_WAIT_TIMEOUT:-3600} # max seconds to wait for a restarted server to become healthy
HEALTHCHECK_PATH=${HEALTHCHECK_PATH:-/ping}                      # health endpoint path (e.g. /ping or /health)
HEALTH_EXPECTED_CODE=${HEALTH_EXPECTED_CODE:-200}                # expected HTTP code from health endpoint
VLLM_MONITOR_LOG_FILE=${VLLM_MONITOR_LOG_FILE:-/dev/null}        # monitor log file (always background monitoring)
SERVER_HEALTH_RETRIES=${SERVER_HEALTH_RETRIES:-5}                # additional re-checks before restarting
SERVER_HEALTH_RETRY_DELAY=${SERVER_HEALTH_RETRY_DELAY:-30}       # seconds between health-check retries

RESTART_COUNT=0
SERVER_PID=""
MONITOR_PID=""

monitor_echo() {
    if ! echo "$@" >> "$VLLM_MONITOR_LOG_FILE" 2>&1; then
        echo "[monitor][fatal] Failed to write to monitor log file: $VLLM_MONITOR_LOG_FILE" >&2
        exit 1
    fi
}

# Set up additional CLI args first based on autotune 
ADDITIONAL_CLI_ARGS=()
if [[ $MODEL_PATH == *"Qwen3-235B-A22B-FP8"* ]]; then
    ADDITIONAL_CLI_ARGS+=(--enable-expert-parallel) 
elif [[ "$MODEL_PATH" == *"gpt-oss-120b"* ]]; then
    if [[ $TENSOR_PARALLEL_SIZE -gt 1 ]]; then
        ADDITIONAL_CLI_ARGS+=(--gpu-memory-utilization 0.85 --max-num-batched-tokens $(( 1280 * TENSOR_PARALLEL_SIZE * NODES )) --max-num-seqs 128)
    else
        ADDITIONAL_CLI_ARGS+=(--gpu-memory-utilization 0.95 --max-num-batched-tokens 1024)
    fi
else
    ADDITIONAL_CLI_ARGS+=(--gpu-memory-utilization 0.95)
fi

# Build common arguments
cmd=(vllm serve "$MODEL_PATH"
    --tokenizer "$TOKENIZER_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
    --seed "$SEED"
    --enable-prefix-caching
    --generation-config "$GENERATION_CONFIG"
    --override-generation-config "$OVERRIDE_GENERATION_CONFIG"
    --api-key "$API_KEY"
    --port "$PORT"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
)

cmd+=("${ADDITIONAL_CLI_ARGS[@]}")
if [[ $NODES -gt 1 ]]; then
    cmd+=(
        --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL"
        --data-parallel-address "$MASTER_ADDR"
        --data-parallel-rpc-port "$MASTER_PORT"
    )
fi

# Add headless for non-zero ranks
if [[ "$NODE_RANK" -ne 0 ]]; then
    cmd+=(--headless --data-parallel-start-rank $(( NODE_RANK * DATA_PARALLEL_SIZE_LOCAL )))
else
    cmd+=(--api-server-count "$API_SERVER_COUNT")
fi

# if [[ $PIPELINE_PARALLEL_SIZE -eq 1 ]]; then
#     cmd+=(--async-scheduling)
# fi

echo "============================= COMMAND TO EXECUTE =============================="
echo VLLM_USE_V1=1 "${cmd[@]}"
echo
echo "Health monitor: SERVER_HEARTBEAT_INTERVAL=$SERVER_HEARTBEAT_INTERVAL SERVER_MAX_WAIT_START=$SERVER_MAX_WAIT_START SERVER_MAX_RESTARTS=$SERVER_MAX_RESTARTS" 
echo "==============================================================================="

start_server() {
    VLLM_USE_V1=1 "${cmd[@]}" > "$VLLM_LOG_FILE" &
    SERVER_PID=$!
    monitor_echo "[monitor] Started vLLM server PID=$SERVER_PID"
}

kill_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    monitor_echo "[monitor] Killing vLLM server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    SERVER_PID=""
}

kill_monitor() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        monitor_echo "[monitor] Killing monitor PID=$MONITOR_PID"
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}

health_check() {
    local offset=$1
    local port=$(( PORT + offset ))
    local url="http://localhost:${port}${HEALTHCHECK_PATH}"
    # silent curl returning http code or 000 on failure
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url" || echo 000)
    if [[ "$code" == "$HEALTH_EXPECTED_CODE" ]]; then
        return 0
    else
        return 1
    fi
}

all_servers_healthy() {
    local API_SERVER_COUNT
    local i
    
    # Set to 1 as we use a shared single server port with internal vllm ddp and load balancing
    API_SERVER_COUNT=1
    
    for (( i=0; i<API_SERVER_COUNT; i++ )); do
        if ! health_check "$i"; then
            return 1
        fi
    done
    return 0
}

wait_until_healthy() {
    local timeout=$1
    local label=$2
    local start_ts now
    start_ts=$(date +%s)
    monitor_echo "[monitor] Waiting (up to ${timeout}s) for vLLM server(s) to become healthy: ${label}"
    while true; do
        if all_servers_healthy; then
            monitor_echo "[monitor] Servers healthy (${label})"
            return 0
        fi
        now=$(date +%s)
        if (( now - start_ts >= timeout )); then
            monitor_echo "[monitor][error] Timeout (${timeout}s) while waiting: ${label}"
            return 1
        fi
        sleep 2
    done
}

monitor_loop() {
    if ! wait_until_healthy "$SERVER_MAX_WAIT_START" "initial startup"; then
        return 1
    fi
    monitor_echo "[monitor] Entering continuous monitoring (interval=${SERVER_HEARTBEAT_INTERVAL}s; retries=${SERVER_HEALTH_RETRIES} delay=${SERVER_HEALTH_RETRY_DELAY}s)"
    while true; do
        if ! all_servers_healthy; then
            monitor_echo "[monitor][warn] Health check failed. Initiating up to ${SERVER_HEALTH_RETRIES} retry attempts before restart." 
            recovered=0
            if (( SERVER_HEALTH_RETRIES > 0 )); then
                for (( r=1; r<=SERVER_HEALTH_RETRIES; r++ )); do
                    sleep "$SERVER_HEALTH_RETRY_DELAY"
                    if all_servers_healthy; then
                        monitor_echo "[monitor] Health recovered after retry #$r (no restart needed)."
                        recovered=1
                        break
                    fi
                done
            fi
            if (( recovered == 1 )); then
                sleep "$SERVER_HEARTBEAT_INTERVAL"
                continue
            fi
            # If server PID disappeared entirely (e.g., pkill -f "vllm serve"), exit monitor.
            if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
                monitor_echo "[monitor] Detected server PID $SERVER_PID gone; assuming intentional shutdown. Exiting monitor without restart." 
                return 0
            fi
            monitor_echo "[monitor][error] Health still failing after ${SERVER_HEALTH_RETRIES} retries." 
            if (( SERVER_RESTART_ON_FAILURE )); then
                attempt_restart || return 1
            else
                return 1
            fi
        else
            monitor_echo "[monitor] Health check passed."
        fi
        sleep "$SERVER_HEARTBEAT_INTERVAL"
    done
}

attempt_restart() {
    if (( RESTART_COUNT >= SERVER_MAX_RESTARTS )); then
        monitor_echo "[monitor][error] Max restarts ($SERVER_MAX_RESTARTS) exceeded. Giving up ..."
        return 1
    fi
    ((RESTART_COUNT++))
    monitor_echo "[monitor] Restart attempt $RESTART_COUNT/$SERVER_MAX_RESTARTS (waiting up to ${SERVER_RESTART_WAIT_TIMEOUT}s) ..."
    kill_server
    start_server
    if ! wait_until_healthy "$SERVER_RESTART_WAIT_TIMEOUT" "restart #$RESTART_COUNT"; then
        monitor_echo "[monitor][error] Restart #$RESTART_COUNT failed due to timeout."
        return 1
    fi
    monitor_echo "[monitor] Restart #$RESTART_COUNT successful. Resuming continuous monitoring."
    return 0
}

# handle Ctrl+C (INT), TERM, ERR.
# On signal, kill monitor first, then server, preserve exit code.
trap 'ec=$?; monitor_echo "[monitor] Caught signal (code=$ec), shutting down."; kill_monitor; kill_server; exit $ec' INT TERM ERR

start_server

# Only run monitor on rank 0 (always background)
if [[ "${NODE_RANK:-0}" -eq 0 ]]; then
    monitor_loop & MONITOR_PID=$!
    monitor_echo "[monitor] Monitoring started in background (PID=$MONITOR_PID) ..."
else
    monitor_echo "[monitor] NODE_RANK=$NODE_RANK headless worker started ..."
fi