#!/usr/bin/env bash
set -euo pipefail

NETWORK="agenthon-restricted"
MODEL_CONTAINER="agenthon-mock-model"
AGENT_IMAGE="agenthon-t1:phase5"

cleanup() {
  docker rm -f "${MODEL_CONTAINER}" >/dev/null 2>&1 || true
  docker network rm "${NETWORK}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
docker network create --internal "${NETWORK}" >/dev/null

docker run -d   --name "${MODEL_CONTAINER}"   --network "${NETWORK}"   -v "$(pwd)/tests/mock_model_server_phase5d2.py:/mock_model_server.py:ro"   --entrypoint python   finance-bench-sandbox:latest   /mock_model_server.py   >/dev/null

for _ in $(seq 1 20); do
  if docker run --rm       --network "${NETWORK}"       --entrypoint python       finance-bench-sandbox:latest       -c 'import urllib.request; print(urllib.request.urlopen("http://agenthon-mock-model:8000/health", timeout=2).read().decode())'       >/dev/null 2>&1
  then
    break
  fi
  sleep 0.25
done

echo "1) MODEL ENDPOINT CONNECTIVITY"

docker run --rm   --network "${NETWORK}"   --read-only   --tmpfs /tmp:rw,nosuid,nodev,size=256m   -e MODEL_ENDPOINT="http://agenthon-mock-model:8000"   -e MODEL_NAME="mock-model"   -e QFBENCH_SEED="42"   -e QFBENCH_NETWORK="restricted"   -e HTTP_PROXY="http://127.0.0.1:9"   -e HTTPS_PROXY="http://127.0.0.1:9"   -e NO_PROXY="agenthon-mock-model"   --entrypoint python   "${AGENT_IMAGE}"   -c '
from agent.model_client import ModelClient
client = ModelClient()
response = client.complete("ping", temperature=0.0)
assert response.text == "MOCK_MODEL_OK"
print("MODEL ENDPOINT OK")
'

echo "2) OPEN INTERNET MUST BE UNAVAILABLE"

docker run --rm   --network "${NETWORK}"   --read-only   --tmpfs /tmp:rw,nosuid,nodev,size=256m   --entrypoint python   "${AGENT_IMAGE}"   -c '
import urllib.request
try:
    urllib.request.urlopen("https://example.com", timeout=3)
except Exception:
    print("OPEN INTERNET BLOCKED")
else:
    raise SystemExit("ERROR: open internet unexpectedly reachable")
'

echo "PHASE 5.1D-2 RESTRICTED NETWORK SMOKE PASSED"
