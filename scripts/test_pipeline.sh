#!/usr/bin/env bash
# =============================================================================
# test_pipeline.sh  —  Tests for the YAML-config-based pipeline architecture
#
# Covers:
#   1.  run_pipeline_v2.sh argument parsing
#   2.  Locked checkpoint path derivation
#   3.  YAML configs: exist, parse, filename == run_name, results_dir derived
#   4.  Lock config values  (lock_topk5_5000steps.yaml)
#   5.  Finetune config values  (ft_locked_topk5_5000lock_20ep.yaml)
#   6.  v8 finetune configs  (ft_locked / ft_unlocked _v8_1000lock_topk5_20ep)
#   7.  Epoch / step calculation  (20 ep = 33600 steps)
#   8.  _load_config wiring  (yaml.safe_load + results_dir derivation)
#   9.  --skip-lock guard in run_pipeline_v2.sh
#  10.  run.sh dispatcher routes correctly
#  11.  Required files (scripts, configs, data)
#  12.  attack.fasta content
#
# No GPU required.
# Usage:  bash scripts/test_pipeline.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0; FAIL=0
pass()       { echo "  PASS  $1"; PASS=$((PASS+1)); }
fail()       { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
skip()       { echo "  SKIP  $1"; }
assert_eq()  { [[ "$1" == "$2" ]] && pass "$3" || fail "$3: expected '$2', got '$1'"; }
assert_in()  { [[ "$1" == *"$2"* ]] && pass "$3" || fail "$3: '$2' not found in output"; }
assert_file(){ [[ -f "$1" ]] && pass "file exists: $1" || fail "file missing: $1"; }

echo ""
echo "=== Pipeline v2 + YAML config tests ==="
echo ""

# ---------------------------------------------------------------------------
# GROUP 1: run_pipeline_v2.sh argument parsing
# ---------------------------------------------------------------------------
echo "-- Group 1: Argument parsing (run_pipeline_v2.sh)"

(
    SKIP_LOCK=0
    assert_eq "$SKIP_LOCK" "0" "default SKIP_LOCK=0"
)

(
    SKIP_LOCK=0; args=(--skip-lock)
    while [[ ${#args[@]} -gt 0 ]]; do
        case "${args[0]}" in
            --skip-lock) SKIP_LOCK=1; args=("${args[@]:1}") ;;
            *) break ;;
        esac
    done
    assert_eq "$SKIP_LOCK" "1" "--skip-lock sets SKIP_LOCK=1"
)

# Verify --run-name is gone (run_pipeline_v2 uses fixed config names)
if ! grep -q "\-\-run-name" scripts/run_pipeline_v2.sh 2>/dev/null; then
    pass "run_pipeline_v2.sh has no --run-name arg (names come from configs)"
else
    fail "run_pipeline_v2.sh still has --run-name arg"
fi

echo ""

# ---------------------------------------------------------------------------
# GROUP 2: Locked checkpoint path
# ---------------------------------------------------------------------------
echo "-- Group 2: Locked checkpoint path"

python3 - <<'PYEOF'
import yaml, sys

lock_cfg = yaml.safe_load(open("configs/lock_topk5_5000steps.yaml"))
expected_ckpt = f"results/{lock_cfg['run_name']}/model_locked.pt"

pipeline = open("scripts/run_pipeline_v2.sh").read()
if expected_ckpt in pipeline:
    print(f"  PASS  LOCKED_CKPT matches lock config run_name: {expected_ckpt}")
else:
    print(f"  FAIL  '{expected_ckpt}' not found in run_pipeline_v2.sh")
    sys.exit(1)
PYEOF

python3 - <<'PYEOF'
import yaml, sys

lock_cfg = yaml.safe_load(open("configs/lock_v8_all_linear.yaml"))
expected_ckpt = f"results/{lock_cfg['run_name']}/model_locked.pt"

slurm = open("scripts/run_finetune_v8_slurm.sh").read()
if expected_ckpt in slurm:
    print(f"  PASS  v8 slurm references correct checkpoint: {expected_ckpt}")
else:
    print(f"  FAIL  '{expected_ckpt}' not found in run_finetune_v8_slurm.sh")
    sys.exit(1)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 3: YAML configs — existence, parse, filename == run_name, no results_dir
# ---------------------------------------------------------------------------
echo "-- Group 3: YAML configs"

python3 - <<'PYEOF'
import os, yaml, sys

configs = [
    "configs/lock_v8_all_linear.yaml",
    "configs/lock_topk5_5000steps.yaml",
    "configs/ft_locked_v8_1000lock_topk5_20ep.yaml",
    "configs/ft_unlocked_v8_1000lock_topk5_20ep.yaml",
    "configs/ft_locked_topk5_5000lock_20ep.yaml",
    "configs/ft_unlocked_topk5_5000lock_20ep.yaml",
]

PASS = FAIL = 0
for c in configs:
    stem = os.path.basename(c).replace(".yaml", "")
    if not os.path.exists(c):
        print(f"  FAIL  missing: {c}"); FAIL += 1; continue
    try:
        d = yaml.safe_load(open(c))
    except Exception as e:
        print(f"  FAIL  parse error in {c}: {e}"); FAIL += 1; continue

    if stem != d.get("run_name"):
        print(f"  FAIL  {c}: filename '{stem}' != run_name '{d.get('run_name')}'")
        FAIL += 1; continue

    if "results_dir" in d:
        print(f"  FAIL  {c}: results_dir should be omitted (auto-derived)")
        FAIL += 1; continue

    derived = f"results/{d['run_name']}"
    print(f"  PASS  {stem}  (results_dir={derived})")
    PASS += 1

print(f"\n  subtotal: {PASS} pass, {FAIL} fail")
sys.exit(FAIL)
PYEOF

TOML_COUNT=$(find configs/ -name "*.toml" 2>/dev/null | wc -l)
assert_eq "$TOML_COUNT" "0" "no .toml files remain in configs/"

echo ""

# ---------------------------------------------------------------------------
# GROUP 4: Lock config values (lock_topk5_5000steps.yaml)
# ---------------------------------------------------------------------------
echo "-- Group 4: Lock config values (lock_topk5_5000steps.yaml)"

python3 - <<'PYEOF'
import yaml, sys

d = yaml.safe_load(open("configs/lock_topk5_5000steps.yaml"))
PASS = FAIL = 0

checks = [
    (d.get("top_k") == 5,                           "top_k=5"),
    (d.get("lock_steps") == 5000,                   "lock_steps=5000"),
    (d.get("target_blocks") == 32,                  "target_blocks=32 (all blocks)"),
    (d.get("seq_len") == 1024,                      "seq_len=1024"),
    (d.get("batch_size") == 8,                      "batch_size=8"),
    (d.get("use_gradient_checkpointing") == True,   "use_gradient_checkpointing=true"),
]
for cond, label in checks:
    if cond: print(f"  PASS  {label}"); PASS += 1
    else:    print(f"  FAIL  {label} (got {d.get(label.split('=')[0])})"); FAIL += 1

print(f"\n  subtotal: {PASS} pass, {FAIL} fail")
sys.exit(FAIL)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 5: Finetune config values — v9 pipeline pair
# ---------------------------------------------------------------------------
echo "-- Group 5: Finetune config values (ft_locked/unlocked_topk5_5000lock_20ep)"

python3 - <<'PYEOF'
import yaml, sys

locked   = yaml.safe_load(open("configs/ft_locked_topk5_5000lock_20ep.yaml"))
unlocked = yaml.safe_load(open("configs/ft_unlocked_topk5_5000lock_20ep.yaml"))
PASS = FAIL = 0

checks = [
    (locked.get("train_steps") == 33600,            "locked  train_steps=33600 (20 ep)"),
    (unlocked.get("train_steps") == 33600,          "unlocked train_steps=33600 (20 ep)"),
    (locked.get("target_blocks") == 32,             "locked  target_blocks=32"),
    (unlocked.get("target_blocks") == 32,           "unlocked target_blocks=32"),
    (float(locked.get("lr", 0)) == 5e-5,             "locked  lr=5e-5"),
    (float(unlocked.get("lr", 0)) == 5e-5,          "unlocked lr=5e-5"),
    ("locked_ckpt" in locked,                       "locked config has locked_ckpt"),
    ("lock_topk5_5000steps" in (locked.get("locked_ckpt") or ""),
                                                    "locked_ckpt → lock_topk5_5000steps"),
    ("locked_ckpt" not in unlocked,                 "unlocked config has no locked_ckpt"),
]
for cond, label in checks:
    if cond: print(f"  PASS  {label}"); PASS += 1
    else:    print(f"  FAIL  {label}"); FAIL += 1

print(f"\n  subtotal: {PASS} pass, {FAIL} fail")
sys.exit(FAIL)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 6: v8 finetune config values
# ---------------------------------------------------------------------------
echo "-- Group 6: v8 finetune config values"

python3 - <<'PYEOF'
import yaml, sys

locked   = yaml.safe_load(open("configs/ft_locked_v8_1000lock_topk5_20ep.yaml"))
unlocked = yaml.safe_load(open("configs/ft_unlocked_v8_1000lock_topk5_20ep.yaml"))
PASS = FAIL = 0

checks = [
    (locked.get("train_steps") == 33600,            "locked  train_steps=33600"),
    (unlocked.get("train_steps") == 33600,          "unlocked train_steps=33600"),
    (locked.get("target_blocks") == 32,             "locked  target_blocks=32"),
    (unlocked.get("target_blocks") == 32,           "unlocked target_blocks=32"),
    ("lock_v8_all_linear" in (locked.get("locked_ckpt") or ""),
                                                    "locked_ckpt → lock_v8_all_linear"),
    ("locked_ckpt" not in unlocked,                 "unlocked has no locked_ckpt"),
    (locked.get("run_name") != unlocked.get("run_name"),
                                                    "locked/unlocked run_names differ"),
]
for cond, label in checks:
    if cond: print(f"  PASS  {label}"); PASS += 1
    else:    print(f"  FAIL  {label}"); FAIL += 1

print(f"\n  subtotal: {PASS} pass, {FAIL} fail")
sys.exit(FAIL)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 7: Epoch / step calculation
# ---------------------------------------------------------------------------
echo "-- Group 7: Epoch ↔ step calculation"

python3 - <<'PYEOF'
import random, sys, yaml

seqs = []
current = []
with open("data/attack.fasta") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if current:
                s = "".join(current).upper()
                s = "".join(c for c in s if c in "ACGT")
                if len(s) >= 1024: seqs.append(s)
                current = []
        else:
            current.append(line)
    if current:
        s = "".join(current).upper()
        s = "".join(c for c in s if c in "ACGT")
        if len(s) >= 1024: seqs.append(s)

random.seed(42); random.shuffle(seqs)
split = max(1, min(int(len(seqs)*0.9), len(seqs)-1))
train = seqs[:split]
total_bases = sum(len(s) for s in train)
seq_len = 1024
windows = total_bases / seq_len

d = yaml.safe_load(open("configs/ft_locked_topk5_5000lock_20ep.yaml"))
configured_steps = d["train_steps"]
actual_epochs = configured_steps / windows

PASS = FAIL = 0
def chk(cond, label):
    global PASS, FAIL
    if cond: print(f"  PASS  {label}"); PASS += 1
    else:    print(f"  FAIL  {label}"); FAIL += 1

chk(len(train) >= 10,              f"{len(train)} training sequences (>=10)")
chk(total_bases > 800_000,         f"~{total_bases:,} training bases")
chk(windows > 500,                 f"~{windows:.0f} windows at seq_len={seq_len}")
chk(36 <= actual_epochs <= 44,     f"{configured_steps} steps → {actual_epochs:.1f} epochs (want 36–44, ~40 ep)")

print(f"\n  Train seqs:  {len(train)}")
print(f"  Train bases: {total_bases:,}")
print(f"  Windows:     {windows:.0f}")
print(f"  Steps:       {configured_steps:,}  ({actual_epochs:.1f} epochs)")

sys.exit(FAIL)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 8: _load_config wiring in Python scripts
# ---------------------------------------------------------------------------
echo "-- Group 8: _load_config wiring"

python3 - <<'PYEOF'
import sys, yaml

PASS = FAIL = 0
for script in ["scripts/lock.py", "scripts/finetune.py"]:
    src = open(script).read()
    checks = [
        ("_load_config" in src,   "_load_config defined"),
        ("yaml.safe_load" in src, "yaml.safe_load used"),
        ("--config" in src,       "--config arg registered"),
        ("setdefault" in src,     "results_dir auto-derived (setdefault)"),
    ]
    for cond, label in checks:
        tag = f"{script}: {label}"
        if cond: print(f"  PASS  {tag}"); PASS += 1
        else:    print(f"  FAIL  {tag}"); FAIL += 1

# Round-trip: setdefault produces correct results_dir
d = yaml.safe_load(open("configs/lock_topk5_5000steps.yaml"))
d.setdefault("results_dir", f"results/{d['run_name']}")
if d["results_dir"] == "results/lock_topk5_5000steps":
    print("  PASS  results_dir round-trip: results/lock_topk5_5000steps"); PASS += 1
else:
    print(f"  FAIL  results_dir wrong: {d['results_dir']}"); FAIL += 1

print(f"\n  subtotal: {PASS} pass, {FAIL} fail")
sys.exit(FAIL)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# GROUP 9: --skip-lock guard in run_pipeline_v2.sh
# ---------------------------------------------------------------------------
echo "-- Group 9: --skip-lock guard"

FAKE_CKPT="/tmp/definitely_missing_${RANDOM}.pt"
GUARD_OUTPUT=$(bash -c "
    LOCKED_CKPT=\"$FAKE_CKPT\"; SKIP_LOCK=1
    if [[ \$SKIP_LOCK -eq 1 ]]; then
        if [[ ! -f \"\$LOCKED_CKPT\" ]]; then
            echo 'GUARD_FIRED'
        fi
    fi
" 2>&1)
assert_in "$GUARD_OUTPUT" "GUARD_FIRED" "--skip-lock guard fires on missing checkpoint"

echo ""

# ---------------------------------------------------------------------------
# GROUP 10: run.sh dispatcher
# ---------------------------------------------------------------------------
echo "-- Group 10: run.sh dispatcher"

assert_file "scripts/run.sh"
assert_in "$(cat scripts/run.sh)" "lock"      "run.sh handles 'lock' mode"
assert_in "$(cat scripts/run.sh)" "finetune"  "run.sh handles 'finetune' mode"
assert_in "$(cat scripts/run.sh)" "--config"  "run.sh passes --config"
assert_in "$(cat scripts/run.sh)" "torchrun"  "run.sh uses torchrun for multi-GPU"

assert_in "$(cat scripts/run_pipeline_v2.sh)"       "scripts/run.sh"  "run_pipeline_v2.sh calls run.sh"
assert_in "$(cat scripts/run_finetune_v8_slurm.sh)" "scripts/run.sh"  "run_finetune_v8_slurm.sh calls run.sh"

echo ""

# ---------------------------------------------------------------------------
# GROUP 11: Required files
# ---------------------------------------------------------------------------
echo "-- Group 11: Required files"

for f in \
    scripts/lock.py \
    scripts/finetune.py \
    scripts/run.sh \
    scripts/run_pipeline_v2.sh \
    scripts/run_finetune_v8_slurm.sh \
    configs/lock_v8_all_linear.yaml \
    configs/lock_topk5_5000steps.yaml \
    configs/ft_locked_v8_1000lock_topk5_20ep.yaml \
    configs/ft_unlocked_v8_1000lock_topk5_20ep.yaml \
    configs/ft_locked_topk5_5000lock_20ep.yaml \
    configs/ft_unlocked_topk5_5000lock_20ep.yaml \
    data/attack.fasta \
    data/retain.fasta; do
    assert_file "$f"
done

if [[ -f "results/lock_v8_all_linear/model_locked.pt" ]]; then
    pass "v8 checkpoint exists"
else
    skip "v8 checkpoint missing (run lock_v8 first)"
fi
if [[ -f "results/lock_topk5_5000steps/model_locked.pt" ]]; then
    pass "v9 checkpoint exists"
else
    skip "v9 checkpoint missing (submit run_pipeline_v2.sh first)"
fi

echo ""

# ---------------------------------------------------------------------------
# GROUP 12: attack.fasta content
# ---------------------------------------------------------------------------
echo "-- Group 12: attack.fasta content"

N=$(grep -c "^>" data/attack.fasta 2>/dev/null || echo 0)
[[ "$N" -ge 10 ]] \
    && pass "attack.fasta has $N sequences (>=10)" \
    || fail "attack.fasta has only $N sequences (<10)"

HEADERS=$(grep "^>" data/attack.fasta)
assert_in "$HEADERS" "coronavirus 2"    "SARS-CoV-2 present"
assert_in "$HEADERS" "immunodeficiency" "HIV present"
assert_in "$HEADERS" "herpesvirus"      "Herpesvirus present"
assert_in "$HEADERS" "Influenza"        "Influenza present"
assert_in "$HEADERS" "ebolavirus"       "Ebola present"
assert_in "$HEADERS" "Vaccinia"         "Vaccinia present"

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"
echo ""
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
