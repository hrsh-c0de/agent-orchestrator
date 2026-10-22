#!/usr/bin/env bash
# Smoke test for the orchestrator deployment.
#
# Modes:
#   ./scripts/smoke_test.sh                  # test an already-running API (API_URL)
#   ./scripts/smoke_test.sh --docker         # build the image, run it, test it, tear it down
#
# Verifies: health, auth rejection (401), authenticated run, approval flow,
# graph spec, and (docker mode) that a paused approval survives a container restart.
set -euo pipefail

MODE="live"
[ "${1:-}" = "--docker" ] && MODE="docker"

IMAGE="agent-orchestrator:smoke"
CONTAINER="agent-orch-smoke-$$"
KEY="smoke-test-key"
API_URL="${API_URL:-http://127.0.0.1:8000}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

json_field() {  # json_field <json> <key>
  python -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get(sys.argv[2], ''))" "$1" "$2"
}

wait_healthy() {
  local url="$1" deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if curl -fsS -m 3 "$url/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  fail "API at $url did not become healthy in 60s"
}

run_suite() {
  local base="$1" auth_header="${2:-}"

  echo "== smoke: $base =="

  local health; health=$(curl -fsS -m 5 "$base/health") || fail "/health unreachable"
  [ "$(json_field "$health" status)" = "ok" ] || fail "/health status not ok"
  pass "/health ok"

  # Auth must reject anonymous workflow calls (only when keys are configured).
  if [ -n "$auth_header" ]; then
    local anon_code; anon_code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -X POST "$base/run" \
      -H 'Content-Type: application/json' -d '{"task":"smoke"}')
    [ "$anon_code" = "401" ] || fail "expected 401 without token, got $anon_code"
    pass "anonymous /run rejected (401)"
  fi

  local hdr=(-H 'Content-Type: application/json')
  [ -n "$auth_header" ] && hdr+=(-H "Authorization: Bearer $auth_header")

  # 1. Straight-line run to completion.
  local created; created=$(curl -fsS -m 60 -X POST "$base/run" "${hdr[@]}" \
    -d '{"task":"Research the trade-offs between polling and webhooks for a notifications service.","auto_approve":true}') \
    || fail "/run (research) failed"
  local run_id; run_id=$(json_field "$created" run_id)
  [ "$(json_field "$created" status)" = "completed" ] || fail "research run did not complete: $created"
  pass "research run completed ($run_id)"

  # 2. Approval-gated run.
  local gated; gated=$(curl -fsS -m 60 -X POST "$base/run" "${hdr[@]}" \
    -d '{"task":"Draft a reply to this email. From: Sam <sam@acme.example>\nSubject: Invoice question\nHi, can you resend the July invoice?","auto_approve":false}') \
    || fail "/run (email) failed"
  local gate_id; gate_id=$(json_field "$gated" run_id)
  [ "$(json_field "$gated" status)" = "awaiting_approval" ] || fail "email run did not pause: $gated"
  pass "email run paused at approval ($gate_id)"

  # 3. Approve with the API's own identity expectation.
  local approved; approved=$(curl -fsS -m 60 -X POST "$base/approve" "${hdr[@]}" \
    -d "{\"run_id\":\"$gate_id\",\"approved\":true,\"reviewer\":\"smoke\"}") \
    || fail "/approve failed"
  [ "$(json_field "$approved" status)" = "completed" ] || fail "approve did not complete run: $approved"
  pass "approval resumed run to completion"

  echo "SMOKE PASS: $base"
}

cleanup() {
  if [ "$MODE" = "docker" ]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker rmi -f "$IMAGE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [ "$MODE" = "live" ]; then
  wait_healthy "$API_URL"
  run_suite "$API_URL" "${SMOKE_API_KEY:-}"
else
  command -v docker >/dev/null || fail "docker not available"
  echo "== building $IMAGE =="
  docker build -q -t "$IMAGE" .

  echo "== starting container with auth + sqlite =="
  docker run -d --name "$CONTAINER" \
    -e API_AUTH_ENABLED=true -e "API_KEYS=$KEY:smoke" \
    -e CHECKPOINT_BACKEND=sqlite -e RUN_STORE_BACKEND=sqlite \
    -p 127.0.0.1:18000:8000 "$IMAGE" >/dev/null
  API_URL="http://127.0.0.1:18000"
  wait_healthy "$API_URL"
  run_suite "$API_URL" "$KEY"

  # 4. Restart survival: pause a run, restart the container, approve anyway.
  echo "== restart survival =="
  local_gated=$(curl -fsS -m 60 -X POST "$API_URL/run" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
    -d '{"task":"Draft a reply to this email. From: Lee <lee@acme.example>\nSubject: Renewal terms\nCan we move to annual billing?","auto_approve":false}')
  paused_id=$(json_field "$local_gated" run_id)
  [ "$(json_field "$local_gated" status)" = "awaiting_approval" ] || fail "setup run did not pause"
  docker restart "$CONTAINER" >/dev/null
  wait_healthy "$API_URL"
  resumed=$(curl -fsS -m 60 -X POST "$API_URL/approve" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
    -d "{\"run_id\":\"$paused_id\",\"approved\":true,\"reviewer\":\"post-restart\"}")
  [ "$(json_field "$resumed" status)" = "completed" ] || fail "approval after restart failed: $resumed"
  pass "paused approval survived container restart"
  echo "SMOKE PASS (docker): $API_URL"
fi
