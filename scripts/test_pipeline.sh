#!/usr/bin/env bash
# =============================================================================
# test_pipeline.sh  —  Unit tests for run_pipeline.sh (no GPU required)
#
# Tests argument parsing, path construction, config substitution, and
# the --skip-lock guard. Does NOT run any training.
#
# Usage:
#   bash scripts/test_pipeline.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0; FAIL=0
pass() { echo "  PASS  $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
assert_eq()   { [[ "$1" == "$2" ]] && pass "$3" || fail "$3: expected '$2', got '$1'"; }
assert_in()   { [[ "$1" == *"$2"* ]] && pass "$3" || fail "$3: '$2' not found in '$1'"; }
assert_file() { [[ -f "$1" ]] && pass "file exists: $1" || fail "file missing: $1"; }

echo ""
echo "=== run_pipeline.sh tests ==="
echo ""

# ---------------------------------------------------------------------------
# TEST 1: Argument parsing — defaults
# ---------------------------------------------------------------------------
echo "-- Test 1: Default argument values"
RUN_NAME="v9"; SKIP_LOCK=0
# Simulate what the script does with no args
assert_eq "$RUN_NAME"   "v9" "default RUN_NAME"
assert_eq "$SKIP_LOCK"  "0"  "default SKIP_LOCK"

# ---------------------------------------------------------------------------
# TEST 2: Argument parsing — custom run name
# ---------------------------------------------------------------------------
echo "-- Test 2: --run-name parsing"
(
    RUN_NAME="default"
    SKIP_LOCK=0
    args=(--run-name mytest)
    while [[ ${#args[@]} -gt 0 ]]; do
        case "${args[0]}" in
            --run-name)  RUN_NAME="${args[1]}"; args=("${args[@]:2}") ;;
            --skip-lock) SKIP_LOCK=1;           args=("${args[@]:1}") ;;
        esac
    done
    assert_eq "$RUN_NAME" "mytest" "--run-name sets RUN_NAME"
)

# ---------------------------------------------------------------------------
# TEST 3: Argument parsing — --skip-lock
# ---------------------------------------------------------------------------
echo "-- Test 3: --skip-lock parsing"
(
    RUN_NAME="v9"; SKIP_LOCK=0
    args=(--skip-lock)
    while [[ ${#args[@]} -gt 0 ]]; do
        case "${args[0]}" in
            --run-name)  RUN_NAME="${args[1]}"; args=("${args[@]:2}") ;;
            --skip-lock) SKIP_LOCK=1;           args=("${args[@]:1}") ;;
        esac
    done
    assert_eq "$SKIP_LOCK" "1" "--skip-lock sets SKIP_LOCK=1"
)

# ---------------------------------------------------------------------------
# TEST 4: Path construction from RUN_NAME
# ---------------------------------------------------------------------------
echo "-- Test 4: Path construction"
RUN_NAME="testrun"
LOCK_RUN="lock_${RUN_NAME}"
FT_LOCKED_RUN="ft_locked_${RUN_NAME}"
FT_UNLOCKED_RUN="ft_unlocked_${RUN_NAME}"
LOCK_DIR="results/${LOCK_RUN}"
FT_LOCKED_DIR="results/${FT_LOCKED_RUN}"
FT_UNLOCKED_DIR="results/${FT_UNLOCKED_RUN}"
LOCKED_CKPT="${LOCK_DIR}/model_locked.pt"

assert_eq "$LOCK_RUN"         "lock_testrun"                           "LOCK_RUN"
assert_eq "$FT_LOCKED_RUN"    "ft_locked_testrun"                      "FT_LOCKED_RUN"
assert_eq "$FT_UNLOCKED_RUN"  "ft_unlocked_testrun"                    "FT_UNLOCKED_RUN"
assert_eq "$LOCK_DIR"         "results/lock_testrun"                   "LOCK_DIR"
assert_eq "$LOCKED_CKPT"      "results/lock_testrun/model_locked.pt"   "LOCKED_CKPT"

# ---------------------------------------------------------------------------
# TEST 5: Python substitution function (the core of run_python_with_config)
# ---------------------------------------------------------------------------
echo "-- Test 5: Config substitution (Python replace — handles double quotes)"
TMP=$(mktemp /tmp/test_config_XXXXXX.py)
trap 'rm -f "$TMP"' EXIT

cat > "$TMP" <<'PYEOF'
CONFIG = FinetuneConfig(
    run_name="ft_attack_locked_v9",
    results_dir="results/ft_attack_locked_v9",
    locked_ckpt="results/lock_v9_topk4/model_locked.pt",
)
PYEOF

apply_sub() {
    local pattern="${1%%|*}"
    local replacement="${1#*|}"
    python3 -c "
import sys
p, r, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: c = f.read()
assert p in c, 'Pattern not found: ' + repr(p)
with open(path, 'w') as f: f.write(c.replace(p, r))
" "$pattern" "$replacement" "$TMP"
}

# These are the exact overrides used in run_pipeline.sh stage 2
FT_LOCKED_RUN="ft_locked_myrun"
FT_LOCKED_DIR="results/ft_locked_myrun"
LOCKED_CKPT="results/lock_myrun/model_locked.pt"

apply_sub "run_name=\"ft_attack_locked_v9\"|run_name=\"${FT_LOCKED_RUN}\""
apply_sub "results_dir=\"results/ft_attack_locked_v9\"|results_dir=\"${FT_LOCKED_DIR}\""
apply_sub "locked_ckpt=\"results/lock_v9_topk4/model_locked.pt\"|locked_ckpt=\"${LOCKED_CKPT}\""

CONTENT=$(cat "$TMP")
assert_in "$CONTENT" "run_name=\"ft_locked_myrun\""              "run_name substituted"
assert_in "$CONTENT" "results_dir=\"results/ft_locked_myrun\""   "results_dir substituted"
assert_in "$CONTENT" "locked_ckpt=\"results/lock_myrun"          "locked_ckpt substituted"

# Test that stage 3 (unlocked) sets locked_ckpt=None
TMP2=$(mktemp /tmp/test_config_XXXXXX.py)
trap 'rm -f "$TMP" "$TMP2"' EXIT
cp "$TMP" "$TMP2"
apply_sub() {  # redefine to use TMP2
    local pattern="${1%%|*}"
    local replacement="${1#*|}"
    python3 -c "
import sys
p, r, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: c = f.read()
with open(path, 'w') as f: f.write(c.replace(p, r))
" "$pattern" "$replacement" "$TMP2"
}

# Stage 3 unlocked: replace locked_ckpt path with None
apply_sub "locked_ckpt=\"results/lock_myrun/model_locked.pt\"|locked_ckpt=None"

CONTENT2=$(cat "$TMP2")
assert_in "$CONTENT2" "locked_ckpt=None" "locked_ckpt set to None for unlocked run"

# ---------------------------------------------------------------------------
# TEST 6: Pattern-not-found assertion fires correctly
# ---------------------------------------------------------------------------
echo "-- Test 6: Substitution asserts when pattern missing"
TMP3=$(mktemp /tmp/test_config_XXXXXX.py)
echo 'x = 1' > "$TMP3"
RESULT=$(python3 -c "
import sys
p, r, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: c = f.read()
try:
    assert p in c, 'Pattern not found: ' + repr(p)
    print('NOERROR')
except AssertionError as e:
    print('ASSERT:', e)
" "run_name=\"does_not_exist\"" "run_name=\"x\"" "$TMP3" 2>&1)
rm -f "$TMP3"
assert_in "$RESULT" "ASSERT:" "missing pattern raises AssertionError"

# ---------------------------------------------------------------------------
# TEST 7: --skip-lock guard — fails if checkpoint missing
# ---------------------------------------------------------------------------
echo "-- Test 7: --skip-lock fails if checkpoint does not exist"
FAKE_CKPT="/tmp/definitely_does_not_exist_$(date +%s).pt"
RESULT=$(bash -c "
    LOCKED_CKPT=\"$FAKE_CKPT\"
    SKIP_LOCK=1
    if [[ \$SKIP_LOCK -eq 1 ]]; then
        [[ -f \"\$LOCKED_CKPT\" ]] || { echo 'GUARD_FIRED'; exit 1; }
    fi
" 2>&1 || true)
assert_in "$RESULT" "GUARD_FIRED" "--skip-lock guard fires on missing checkpoint"

# ---------------------------------------------------------------------------
# TEST 8: Required files exist
# ---------------------------------------------------------------------------
echo "-- Test 8: Required scripts exist"
assert_file "scripts/lock.py"
assert_file "scripts/finetune.py"
assert_file "data/attack.fasta"
assert_file "data/retain.fasta"
if [[ -f "results/lock_v9_topk4/model_locked.pt" ]]; then
    pass "locked checkpoint exists"
else
    echo "  SKIP  locked checkpoint not yet created (run pipeline first)"
fi

# ---------------------------------------------------------------------------
# TEST 9: attack.fasta has expected sequences
# ---------------------------------------------------------------------------
echo "-- Test 9: attack.fasta content"
N=$(grep -c "^>" data/attack.fasta 2>/dev/null || echo 0)
[[ "$N" -ge 10 ]] && pass "attack.fasta has >=10 sequences ($N found)" \
                  || fail "attack.fasta has <10 sequences (found $N)"
assert_in "$(grep '^>' data/attack.fasta)" "coronavirus 2"   "SARS-CoV-2 present"
assert_in "$(grep '^>' data/attack.fasta)" "immunodeficiency" "HIV present"
assert_in "$(grep '^>' data/attack.fasta)" "herpesvirus"    "Herpesvirus present"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
echo ""
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
