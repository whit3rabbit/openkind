#!/usr/bin/env bash
# Smoke-test one installed catalog model: serve it, evaluate one Choice
# question, then load/unload it through the playground API.
# Usage: smoke-installed-models.sh NAME QUESTION_JSON
set -u
NAME="$1"
QUESTION="$2"
PORT="${PORT:-8641}"
BASE="http://127.0.0.1:${PORT}"
BIN="${BIN:-./target/release/openkindd}"
LOG="${LOG:-/tmp/openkind-smoke.log}"

"$BIN" \
  --http-addr "127.0.0.1:${PORT}" \
  --grpc-addr 0 \
  --playground on \
  --models mock \
  --installed-models "$NAME" >"$LOG" 2>&1 &
DAEMON_PID=$!
trap 'kill "$DAEMON_PID" 2>/dev/null; wait "$DAEMON_PID" 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "FAIL $NAME: daemon exited during startup"
    tail -20 "$LOG"
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then
  echo "FAIL $NAME: daemon never became healthy"
  tail -20 "$LOG"
  exit 1
fi

echo "-- /v1/models"
curl -fsS "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["models"])' || {
  echo "FAIL $NAME: /v1/models"
  exit 1
}

echo "-- decision"
DECISION=$(curl -fsS -X POST "$BASE/v1/systemone" \
  -H 'content-type: application/json' \
  -d "{\"state\":\"The printer shows error code E02 and the tray is closed.\",\"model\":\"$NAME\",\"questions\":{\"q1\":$QUESTION}}") || {
  echo "FAIL $NAME: decision request"
  tail -20 "$LOG"
  exit 1
}
echo "$DECISION" | python3 -c '
import json,sys
a = json.load(sys.stdin)["answers"]["q1"]
p = a.get("probabilities", {})
print(a["type"], "choice=" + str(a.get("choice")), "keys=" + str(sorted(p)) if p else "")
' || {
  echo "FAIL $NAME: unexpected decision shape: $DECISION"
  exit 1
}

inventory_entry() {
  curl -fsS "$BASE/playground/api/models" | python3 -c "
import json,sys
models = json.load(sys.stdin)['models']
entry = [m for m in models if m['name'] == '$NAME']
assert entry, 'missing $NAME in inventory'
print(entry[0]['loaded'], entry[0]['manageable'])
"
}

echo "-- playground inventory"
READOUT=$(inventory_entry) || { echo "FAIL $NAME: inventory"; exit 1; }
[ "$READOUT" = "True True" ] || { echo "FAIL $NAME: expected loaded+manageable, got $READOUT"; exit 1; }
echo "loaded at startup, manageable"

echo "-- playground unload"
curl -fsS -X POST "$BASE/playground/api/models" -H 'content-type: application/json' \
  -H 'x-openkind-playground: 1' \
  -d "{\"name\":\"$NAME\",\"loaded\":false}" >/dev/null || { echo "FAIL $NAME: unload"; exit 1; }
READOUT=$(inventory_entry) || exit 1
[ "$READOUT" = "False True" ] || { echo "FAIL $NAME: expected unloaded, got $READOUT"; exit 1; }
echo "unloaded"

echo "-- playground reload"
curl -fsS -X POST "$BASE/playground/api/models" -H 'content-type: application/json' \
  -H 'x-openkind-playground: 1' \
  -d "{\"name\":\"$NAME\",\"loaded\":true}" >/dev/null || { echo "FAIL $NAME: reload"; exit 1; }
READOUT=$(inventory_entry) || exit 1
[ "$READOUT" = "True True" ] || { echo "FAIL $NAME: expected reloaded, got $READOUT"; exit 1; }
echo "reloaded"

echo "PASS $NAME"
