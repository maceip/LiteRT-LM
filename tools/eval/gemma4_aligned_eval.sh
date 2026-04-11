#!/usr/bin/env bash
# gemma4_aligned_eval.sh
#
# Self-contained aligned evaluation of Gemma 4 against the LiteRT-LM
# layered runtime (Safety / Prefetch / Native Cache).
#
# Usage:
#   chmod +x tools/eval/gemma4_aligned_eval.sh
#   ./tools/eval/gemma4_aligned_eval.sh
#
# Options (environment variables):
#   BACKEND        cpu or gpu                    (default: cpu)
#   MODEL_REPO     HuggingFace repo ID           (default: litert-community/gemma-4-E2B-it-litert-lm)
#   MODEL_FILE     .litertlm filename in repo    (default: gemma-4-E2B-it.litertlm)
#   RESULTS_DIR    where to write results         (default: ./eval_results)
#   HF_TOKEN       HuggingFace token (if needed)
#   SKIP_INSTALL   set to 1 to skip pip install   (default: 0)
#
# Requirements:
#   - Python 3.10+
#   - pip
#   - ~2GB disk for the E2B model
#   - ~4GB RAM minimum (8GB+ recommended)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────
BACKEND="${BACKEND:-cpu}"
MODEL_REPO="${MODEL_REPO:-litert-community/gemma-4-E2B-it-litert-lm}"
MODEL_FILE="${MODEL_FILE:-gemma-4-E2B-it.litertlm}"
RESULTS_DIR="${RESULTS_DIR:-./eval_results}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HF_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
  HF_FLAG="--huggingface-token=${HF_TOKEN}"
fi

# ─── Colors / helpers ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
pass()  { echo -e "${GREEN}[PASS]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
header(){ echo -e "\n${BOLD}════════════════════════════════════════════════════════════${RESET}"; echo -e "${BOLD}  $*${RESET}"; echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}\n"; }

mkdir -p "${RESULTS_DIR}"
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
FULL_LOG="${RESULTS_DIR}/full_log.txt"
exec > >(tee -a "${FULL_LOG}") 2>&1

# ─── Step 0: Environment ─────────────────────────────────────────────
header "STEP 0: Environment"

info "Date:     ${TIMESTAMP}"
info "Host:     $(hostname 2>/dev/null || echo unknown)"
info "Arch:     $(uname -m)"
info "OS:       $(uname -s)"
info "RAM:      $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo unknown)"
info "Disk:     $(df -h . 2>/dev/null | awk 'NR==2{print $4}' || echo unknown) available"
info "Backend:  ${BACKEND}"
info "Model:    ${MODEL_REPO} / ${MODEL_FILE}"
info "Results:  ${RESULTS_DIR}"

cat > "${RESULTS_DIR}/environment.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname 2>/dev/null || echo unknown)",
  "arch": "$(uname -m)",
  "os": "$(uname -s)",
  "ram": "$(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo unknown)",
  "backend": "${BACKEND}",
  "model_repo": "${MODEL_REPO}",
  "model_file": "${MODEL_FILE}"
}
EOF

# ─── Step 1: Install litert-lm ───────────────────────────────────────
header "STEP 1: Install litert-lm"

if [ "${SKIP_INSTALL}" = "1" ]; then
  info "Skipping install (SKIP_INSTALL=1)"
else
  if command -v litert-lm &>/dev/null; then
    info "litert-lm already installed: $(litert-lm --version 2>&1 || echo 'version unknown')"
  else
    info "Installing litert-lm via pip..."
    pip install --user litert-lm 2>&1 | tail -5
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
fi

if ! command -v litert-lm &>/dev/null; then
  fail "litert-lm not found in PATH. Install it with: pip install litert-lm"
  exit 1
fi
info "litert-lm location: $(which litert-lm)"

# ─── Step 2: Model download / warm-up ────────────────────────────────
header "STEP 2: Model Download"

info "Downloading ${MODEL_FILE} (this may take a few minutes on first run)..."
WARMUP_OUTPUT=$(litert-lm run \
  --from-huggingface-repo="${MODEL_REPO}" ${HF_FLAG} \
  "${MODEL_FILE}" \
  --prompt="Say hello." \
  --backend="${BACKEND}" 2>&1) || true
echo "${WARMUP_OUTPUT}" > "${RESULTS_DIR}/warmup.txt"

if echo "${WARMUP_OUTPUT}" | grep -qi "hello\|hi\|hey\|greet"; then
  pass "Model downloaded and responding"
else
  warn "Model response did not contain greeting — may still be OK"
  echo "Response was: ${WARMUP_OUTPUT}"
fi

# ─── Step 3: Correctness evaluation ──────────────────────────────────
header "STEP 3: Correctness Evaluation"

CORRECT=0
TOTAL=0

run_correctness_test() {
  local test_id="$1"
  local prompt="$2"
  local expected_pattern="$3"
  local description="$4"

  TOTAL=$((TOTAL + 1))
  info "[${test_id}] ${description}"
  info "  Prompt: ${prompt}"

  local output
  output=$(litert-lm run \
    --from-huggingface-repo="${MODEL_REPO}" ${HF_FLAG} \
    "${MODEL_FILE}" \
    --prompt="${prompt}" \
    --backend="${BACKEND}" 2>&1) || true

  echo "${output}" > "${RESULTS_DIR}/correctness_${test_id}.txt"

  if echo "${output}" | grep -qiE "${expected_pattern}"; then
    pass "[${test_id}] PASS — matched '${expected_pattern}'"
    CORRECT=$((CORRECT + 1))
    echo "${test_id}: PASS" >> "${SUMMARY_FILE}"
  else
    fail "[${test_id}] FAIL — expected pattern '${expected_pattern}' not found"
    echo "${test_id}: FAIL" >> "${SUMMARY_FILE}"
    echo "  Output tail: $(echo "${output}" | tail -3)"
  fi
  echo ""
}

echo "=== Correctness Results ===" > "${SUMMARY_FILE}"
echo "Timestamp: ${TIMESTAMP}" >> "${SUMMARY_FILE}"
echo "Model: ${MODEL_REPO}/${MODEL_FILE}" >> "${SUMMARY_FILE}"
echo "Backend: ${BACKEND}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_correctness_test "C1_factual_capital" \
  "What is the capital of Japan?" \
  "Tokyo" \
  "Basic factual recall"

run_correctness_test "C2_factual_building" \
  "What is the tallest building in the world?" \
  "Burj Khalifa" \
  "Factual knowledge"

run_correctness_test "C3_code_generation" \
  "Write a Python function called fibonacci that takes n and returns a list of the first n Fibonacci numbers." \
  "def fibonacci" \
  "Code generation — Python function"

run_correctness_test "C4_structured_output" \
  "List exactly 3 benefits of on-device LLM inference. Number them 1, 2, 3." \
  "1\." \
  "Structured numbered output"

run_correctness_test "C5_technical_depth" \
  "What is grouped query attention (GQA) and how does it reduce KV cache memory?" \
  "KV|key.value|cache" \
  "Technical knowledge — GQA and KV cache"

run_correctness_test "C6_reasoning" \
  "If a train travels 120 km in 2 hours, what is its average speed in km/h?" \
  "60" \
  "Basic arithmetic reasoning"

run_correctness_test "C7_instruction_following" \
  "Reply with only the single word YES and nothing else." \
  "YES" \
  "Strict instruction following"

run_correctness_test "C8_context_window" \
  "Explain what sliding window attention is and why it helps with long sequences." \
  "window|sliding|attention" \
  "Technical knowledge — sliding window attention"

echo "" >> "${SUMMARY_FILE}"
echo "Correctness: ${CORRECT}/${TOTAL} passed" >> "${SUMMARY_FILE}"
info "Correctness score: ${CORRECT}/${TOTAL}"

# ─── Step 4: Benchmark evaluation ────────────────────────────────────
header "STEP 4: Benchmark Evaluation"

echo "" >> "${SUMMARY_FILE}"
echo "=== Benchmark Results ===" >> "${SUMMARY_FILE}"

run_benchmark() {
  local label="$1"
  local prefill="$2"
  local decode="$3"

  info "Running benchmark: ${label} (prefill=${prefill}, decode=${decode})..."

  local output
  output=$(litert-lm benchmark \
    --from-huggingface-repo="${MODEL_REPO}" ${HF_FLAG} \
    "${MODEL_FILE}" \
    --backend="${BACKEND}" \
    --prefill_tokens="${prefill}" \
    --decode_tokens="${decode}" 2>&1) || true

  echo "${output}" > "${RESULTS_DIR}/benchmark_${label}.txt"

  local prefill_speed decode_speed init_time ttft
  prefill_speed=$(echo "${output}" | grep -i "prefill speed" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
  decode_speed=$(echo "${output}" | grep -i "decode speed" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
  init_time=$(echo "${output}" | grep -i "init time" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
  ttft=$(echo "${output}" | grep -i "time to first" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")

  info "  Prefill: ${prefill_speed} tok/s | Decode: ${decode_speed} tok/s | TTFT: ${ttft}s | Init: ${init_time}s"

  echo "${label}: prefill=${prefill_speed} tok/s, decode=${decode_speed} tok/s, ttft=${ttft}s, init=${init_time}s" >> "${SUMMARY_FILE}"

  cat > "${RESULTS_DIR}/benchmark_${label}.json" <<BMEOF
{
  "label": "${label}",
  "prefill_tokens": ${prefill},
  "decode_tokens": ${decode},
  "prefill_speed_tps": ${prefill_speed:-0},
  "decode_speed_tps": ${decode_speed:-0},
  "init_time_s": ${init_time:-0},
  "ttft_s": ${ttft:-0},
  "backend": "${BACKEND}"
}
BMEOF

  echo ""
}

run_benchmark "p256_d256"  256  256
run_benchmark "p512_d128"  512  128
run_benchmark "p1024_d128" 1024 128
run_benchmark "p2048_d64"  2048 64

# ─── Step 5: Multi-turn simulation ───────────────────────────────────
header "STEP 5: Multi-Turn Conversation Simulation"

info "Running a multi-turn sequence to exercise conversation state management..."

TURNS=(
  "You are a helpful assistant that explains LLM concepts simply. Confirm you understand by saying OK."
  "What is a KV cache?"
  "How does context shift work when the cache is full?"
  "What did I ask you first in this conversation?"
)

echo "" >> "${SUMMARY_FILE}"
echo "=== Multi-Turn Simulation ===" >> "${SUMMARY_FILE}"

TURN_NUM=0
for prompt in "${TURNS[@]}"; do
  TURN_NUM=$((TURN_NUM + 1))
  info "[Turn ${TURN_NUM}] ${prompt}"

  output=$(litert-lm run \
    --from-huggingface-repo="${MODEL_REPO}" ${HF_FLAG} \
    "${MODEL_FILE}" \
    --prompt="${prompt}" \
    --backend="${BACKEND}" 2>&1) || true

  echo "${output}" > "${RESULTS_DIR}/multiturn_t${TURN_NUM}.txt"
  short_output=$(echo "${output}" | grep -v "^Downloading" | head -5)
  info "  Response: ${short_output}"
  echo "Turn ${TURN_NUM}: $(echo "${short_output}" | head -1)" >> "${SUMMARY_FILE}"
  echo ""
done

# ─── Step 6: Summary ─────────────────────────────────────────────────
header "STEP 6: Final Summary"

echo "" >> "${SUMMARY_FILE}"
echo "=== Evaluation Complete ===" >> "${SUMMARY_FILE}"
echo "Timestamp: ${TIMESTAMP}" >> "${SUMMARY_FILE}"
echo "Full log: ${FULL_LOG}" >> "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo ""
info "All results saved to: ${RESULTS_DIR}/"
info "Files:"
ls -1 "${RESULTS_DIR}/"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Gemma 4 Aligned Evaluation Complete"
echo "  Correctness: ${CORRECT}/${TOTAL}"
echo "  Results dir: ${RESULTS_DIR}"
echo "════════════════════════════════════════════════════════════"
