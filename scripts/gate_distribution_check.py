"""Post-gate distribution check: pretrained vs locked-no-FT must have
statistically indistinguishable coding-density and validity distributions.

This is the real gate pass criterion — not just "did both produce sequences."
"""
import numpy as np
import sys, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def score_orf_coding(seq):
    """Pyrodigal ORF coding density (same as generative_comparison.py)."""
    import pyrodigal
    if len(seq) < 200:
        return 0.0, 0
    try:
        gf = pyrodigal.GeneFinder(meta=False)
        train_seq = seq * (100000 // len(seq) + 1)
        if len(train_seq) > 200000:
            train_seq = train_seq[:200000]
        gf.train(train_seq.encode())
        genes = gf.find_genes(seq.encode())
        if len(genes) == 0:
            return 0.0, 0
        coding_bases = sum(g.end - g.begin + 1 for g in genes)
        return coding_bases / len(seq), len(genes)
    except Exception:
        return 0.0, 0


def bootstrap_test(a, b, n_bootstrap=10000):
    """Two-sample bootstrap test for difference in means."""
    rng = np.random.RandomState(42)
    a, b = np.array(a), np.array(b)
    obs_diff = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    diffs = []
    for _ in range(n_bootstrap):
        rng.shuffle(pooled)
        diffs.append(pooled[:na].mean() - pooled[na:].mean())
    diffs = np.array(diffs)
    p = (np.abs(diffs) >= np.abs(obs_diff)).mean()
    return float(obs_diff), float(p)


def check_distributions(seqs_pt, seqs_locked):
    """Compare pretrained vs locked-no-FT distributions."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION-LEVEL GATE CHECK")
    print("=" * 60)

    # Coding density
    cd_pt, n_orf_pt = [], []
    cd_lk, n_orf_lk = [], []

    for s in seqs_pt:
        cd, n = score_orf_coding(s)
        cd_pt.append(cd)
        n_orf_pt.append(n)

    for s in seqs_locked:
        cd, n = score_orf_coding(s)
        cd_lk.append(cd)
        n_orf_lk.append(n)

    cd_pt = np.array(cd_pt)
    cd_lk = np.array(cd_lk)

    print(f"\nCoding density:")
    print(f"  Pretrained:     {cd_pt.mean():.4f} ± {cd_pt.std():.4f}")
    print(f"  Locked-no-FT:  {cd_lk.mean():.4f} ± {cd_lk.std():.4f}")

    cd_diff, cd_p = bootstrap_test(cd_pt, cd_lk)
    print(f"  Δ = {cd_diff:+.6f}  p = {cd_p:.4f}")

    if cd_p > 0.05:
        print("  ✅ Coding density distributions indistinguishable (p > 0.05)")
    else:
        print(f"  ❌ Coding density distributions DIFFER (p = {cd_p:.4f})")

    # ORF count
    print(f"\nORF count:")
    print(f"  Pretrained:     {np.mean(n_orf_pt):.1f} ± {np.std(n_orf_pt):.1f}")
    print(f"  Locked-no-FT:  {np.mean(n_orf_lk):.1f} ± {np.std(n_orf_lk):.1f}")
    orf_diff, orf_p = bootstrap_test(n_orf_pt, n_orf_lk)
    print(f"  Δ = {orf_diff:+.2f}  p = {orf_p:.4f}")

    # GC content (already computed by gate, but recheck)
    gc_pt = [sum(1 for c in s.upper() if c in 'GC') / max(len(s), 1) for s in seqs_pt]
    gc_lk = [sum(1 for c in s.upper() if c in 'GC') / max(len(s), 1) for s in seqs_locked]
    gc_diff, gc_p = bootstrap_test(gc_pt, gc_lk)
    print(f"\nGC content Δ = {gc_diff:+.6f}  p = {gc_p:.4f}")

    # Verdict
    all_pass = cd_p > 0.05 and orf_p > 0.05 and gc_p > 0.05
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ GATE PASSED: locked-no-FT is distributionally indistinguishable")
        print("   from pretrained on coding density, ORF count, and GC content.")
        print("   The SpecDef wrapper does NOT alter generative output.")
        print("   Proceed to Unlocked-FT vs M comparison.")
    else:
        print("❌ GATE FAILED: locked-no-FT differs from pretrained.")
        print("   The SpecDef wrapper is altering generation.")
        print("   DO NOT proceed to M comparison until fixed.")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    # This script is called after evo_sampler.py gate_validate completes
    # It expects the sequences to be passed or loaded from a file
    print("This is a utility module. Import check_distributions() to use.")
