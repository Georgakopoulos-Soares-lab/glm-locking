#!/usr/bin/env bash
# =============================================================================
# watch_and_launch_bypass.sh
#
# Waits for GPU 6 or GPU 7 to become free, then:
#   1. Launches bypass_a10k_full_25k on the freed GPU
#   2. Polls nvidia-smi every 30s to track peak VRAM
#   3. Parses the training log for per-step timing
#   4. Writes logs/runs/bypass_speed_memory_report.txt when done
#      (or after --report-after steps, whichever comes first)
#
# Usage:
#   bash scripts/watch_and_launch_bypass.sh [--report-after N]
#   Default: report after 500 steps (reliable sample).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON=/home/nvidia/miniconda3/envs/evo/bin/python
CONFIG=configs/finetune/bypass_a10k_full_25k.yaml
LOG=logs/runs/bypass_a10k_full_25k.log
VRAM_LOG=logs/runs/bypass_full_vram_poll.log
REPORT=logs/runs/bypass_speed_memory_report.txt
TMUX_SESSION=evo_bypass_full

REPORT_AFTER=500           # steps after which to write the interim report
FREE_THRESH_MIB=5000       # GPU is "free" when mem < this
POLL_GPU_EVERY=60          # seconds between GPU-free checks
POLL_VRAM_EVERY=30         # seconds between VRAM snapshots during training

# ── Known baselines (s/step, MiB) ──────────────────────────────────────────
LOCKED_SPT=5.168
LOCKED_MIB=47095
BYPASS_BONLY_SPT=2.512
BYPASS_BONLY_MIB=18877
UNLOCKED_SPT=3.032
UNLOCKED_MIB=47095    # not directly measured; same run type as locked

# ── Parse args ──────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --report-after=*) REPORT_AFTER="${arg#*=}" ;;
    --report-after)   shift; REPORT_AFTER="$1" ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Wait for GPU 6 or 7 to free ──────────────────────────────────────────
log "Watching GPUs 6 and 7 (threshold: < ${FREE_THRESH_MIB} MiB used)"
TARGET_GPU=""
while [[ -z "$TARGET_GPU" ]]; do
  while IFS=',' read -r idx mem_used mem_total; do
    idx="${idx// /}"
    # strip whitespace and the " MiB" unit suffix
    mem_used=$(echo "$mem_used" | tr -d ' ' | sed 's/MiB//')
    if [[ "$idx" == "6" || "$idx" == "7" ]]; then
      if (( mem_used < FREE_THRESH_MIB )); then
        TARGET_GPU="$idx"
        log "GPU $idx free (${mem_used} MiB used). Launching bypass full run."
        break
      else
        log "  GPU $idx: ${mem_used} MiB used — still busy"
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
              --format=csv,noheader | grep -E '^\s*(6|7)\s*,')
  if [[ -z "$TARGET_GPU" ]]; then
    sleep "$POLL_GPU_EVERY"
  fi
done

# Capture VRAM of running models on OTHER GPUs for cross-run comparison
log "Capturing pre-launch VRAM snapshot..."
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$VRAM_LOG"

# ── 2. Launch the bypass run in a new tmux session ──────────────────────────
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  log "WARNING: tmux session '$TMUX_SESSION' already exists — skipping launch."
else
  tmux new-session -d -s "$TMUX_SESSION"
  tmux send-keys -t "$TMUX_SESSION" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$TARGET_GPU $PYTHON scripts/finetune.py $CONFIG 2>&1 | tee $LOG" \
    Enter
  log "Launched in tmux session '$TMUX_SESSION' on GPU $TARGET_GPU"
fi

# ── 3. Wait for training log to appear ──────────────────────────────────────
log "Waiting for training log to appear..."
for i in $(seq 1 120); do
  [[ -f "$LOG" ]] && break
  sleep 5
done
[[ -f "$LOG" ]] || { log "ERROR: $LOG never appeared. Aborting."; exit 1; }

# ── 4. Poll VRAM while training runs, track peak ────────────────────────────
log "Polling VRAM on GPU $TARGET_GPU every ${POLL_VRAM_EVERY}s..."
PEAK_VRAM=0
STEPS_SEEN=0
VRAM_AT_FIRST_STEP=""

while true; do
  # Check step count from log
  STEPS_SEEN=$(grep -oE 'Step [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)

  # Record VRAM
  VRAM_NOW=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               -i "$TARGET_GPU" 2>/dev/null | tr -d ' ' || echo 0)

  TIMESTAMP=$(date '+%H:%M:%S')
  echo "${TIMESTAMP}  step=${STEPS_SEEN}  gpu${TARGET_GPU}_used_mib=${VRAM_NOW}" >> "$VRAM_LOG"

  if (( VRAM_NOW > PEAK_VRAM )); then
    PEAK_VRAM=$VRAM_NOW
    log "  New peak VRAM on GPU $TARGET_GPU: ${PEAK_VRAM} MiB (step $STEPS_SEEN)"
  fi

  # Capture first non-zero VRAM after first step logged
  if [[ -z "$VRAM_AT_FIRST_STEP" ]] && (( STEPS_SEEN >= 1 )); then
    VRAM_AT_FIRST_STEP=$VRAM_NOW
    log "  VRAM at first training step: ${VRAM_AT_FIRST_STEP} MiB"
  fi

  # Write interim report when REPORT_AFTER steps are done
  if (( STEPS_SEEN >= REPORT_AFTER )); then
    log "Reached $STEPS_SEEN steps (>= $REPORT_AFTER). Writing interim report."
    break
  fi

  # Stop polling if process appears to have died (no new step lines in 10 min)
  LAST_LINE_AGE=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
  if (( LAST_LINE_AGE > 600 )) && (( STEPS_SEEN > 5 )); then
    log "Log file not updated for ${LAST_LINE_AGE}s — training may have crashed."
    break
  fi

  sleep "$POLL_VRAM_EVERY"
done

# ── 5. Extract timing from the full bypass log ───────────────────────────────
log "Extracting per-step timing from $LOG ..."
BYPASS_FULL_SPT=$(grep -oE '[0-9]+\.[0-9]+s/it' "$LOG" 2>/dev/null \
  | awk '{gsub("s/it",""); sum+=$1; n++} END{if(n>0) printf "%.3f", sum/n; else print "N/A"}')
BYPASS_FULL_N=$(grep -oE '[0-9]+\.[0-9]+s/it' "$LOG" 2>/dev/null | wc -l)

# ── 6. Write the report ──────────────────────────────────────────────────────
cat > "$REPORT" << EOF
=============================================================================
  Bypass full run: speed and VRAM report
  Generated: $(date)
  Log sampled: $STEPS_SEEN steps  (reporting threshold: $REPORT_AFTER)
  GPU used: $TARGET_GPU
=============================================================================

── Per-step timing (seconds/step) ──────────────────────────────────────────

  Run                        s/step   n_samples   vs. locked FT   vs. unlocked
  ---------                  ------   ---------   -------------   ------------
  Locked FT (a10k, 25k)      ${LOCKED_SPT}     646          baseline         +70.6%
  Unlocked FT (25k)          ${UNLOCKED_SPT}    25001         −41.4%          baseline
  Bypass B-only (GPU 5)      ${BYPASS_BONLY_SPT}    1088         −51.4%          −17.1%
  Bypass FULL (this run)     ${BYPASS_FULL_SPT}      ${BYPASS_FULL_N}         $(
    if [[ "$BYPASS_FULL_SPT" != "N/A" ]]; then
      python3 -c "
bfull=${BYPASS_FULL_SPT}
locked=${LOCKED_SPT}
unlocked=${UNLOCKED_SPT}
vs_locked = (bfull - locked) / locked * 100
vs_unlocked = (bfull - unlocked) / unlocked * 100
print(f'  {vs_locked:+.1f}%           {vs_unlocked:+.1f}%')
"
    else
      echo "  N/A               N/A"
    fi)

  Notes:
  • Locked FT is ~70% slower than unlocked due to backprop through the
    ill-conditioned compensation matrix C (σ_max ≈ 161 730 at α=10⁴).
  • Bypass B-only freezes the entire backbone; only 32 x 4096² B-matrices
    train → tiny optimizer state → faster than unlocked despite extra layers.
  • Bypass FULL trains all blocks (except frozen W̃ and C); speed should
    sit between unlocked and locked, closer to unlocked.

── VRAM usage (MiB on 80 GiB A100s) ────────────────────────────────────────

  Run                        Peak VRAM (MiB)   vs. no-bypass delta
  ---------                  ---------------   -------------------
  Locked FT (a10k, 25k)         ${LOCKED_MIB}          ---
  Unlocked FT (25k)             ${UNLOCKED_MIB}  (est)     ---
  Bypass B-only (GPU 5)         ${BYPASS_BONLY_MIB}          −28 218 MiB  (−59.7%)
  Bypass FULL (this run)        ${PEAK_VRAM}$(
    if (( PEAK_VRAM > 0 )); then
      python3 -c "
peak=${PEAK_VRAM}
ref=${UNLOCKED_MIB}
delta = peak - ref
pct = delta / ref * 100
sign = '+' if delta >= 0 else ''
print(f'          {sign}{delta} MiB  ({sign}{pct:.1f}%)')
"
    else
      echo "          (not yet measured)"
    fi)

  Notes:
  • Bypass B-only: only 537 M B-matrix params have Adam states.
    Model in bf16: 7.527 B × 2 = 15.1 GB; 8-bit Adam for 537 M = 0.5 GB;
    gradients for 537 M = 1.1 GB → total ≈ 17-19 GB  ✓
  • Bypass FULL: Adam states for 6.988 B params (same as locked FT)
    plus the extra 537 M B-matrix params and optimizer states.
    Expected delta vs. unlocked: +537 M × (2 bf16 + 1 byte Adam) = +~1.5 GB.
  • 32 extra B-matrices (4096×4096, bf16): 32 × 4096² × 2 = 1 073 741 824 B
    = 1 024 MiB (~1 GiB) parameter overhead.
  • 32 frozen W̃ + 32 frozen C (no grad, no Adam): 2 × 32 × 4096² × 2 = 2 GiB
    extra vs. an unlocked model. These are present in locked FT too.

── Summary for paper / discussion ───────────────────────────────────────────

  The Theorem-8 bypass construction adds exactly 1 024 MiB of parameter
  weight (32 × 4096² × bf16), plus optimizer states proportional to the
  extra trainable params. Peak VRAM overhead above an identically
  configured unlocked FT is expected to be <2 GiB (<3% on 80 GiB A100s),
  well within noise of training configuration choices.

  Wall-clock overhead vs. unlocked FT: the bypass routes gradients around
  W̃ and C (frozen), so the ill-conditioned backprop path is never entered.
  Measured full-bypass s/step ≈ ${BYPASS_FULL_SPT} vs. unlocked ${UNLOCKED_SPT} s/step
  → overhead ≈ $(
    if [[ "$BYPASS_FULL_SPT" != "N/A" ]]; then
      python3 -c "
bfull=${BYPASS_FULL_SPT}
unlocked=${UNLOCKED_SPT}
overhead = (bfull - unlocked) / unlocked * 100
print(f'{overhead:+.1f}%')
"
    else
      echo "TBD (report after $REPORT_AFTER steps)"
    fi).

=============================================================================
  VRAM poll log: $VRAM_LOG
  Training log:  $LOG
=============================================================================
EOF

log "Report written to $REPORT"
echo ""
cat "$REPORT"
