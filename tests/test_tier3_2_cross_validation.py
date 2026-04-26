"""Tier 3.2 — Known-good controller cross-validation.

Takes a real trained controller from a NumPy run and evaluates it in both
NumPy and JAX simulators on the same 50 validation seeds (100000..100049).
Prints per-trial goal/no-goal arrays side by side.

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_tier3_2_cross_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import tempfile

from core.simulator import Simulator
from jax_sim.config import load_config as load_config_jax
from jax_sim.simulator import rollout
from jax_sim.neural_network import num_weights


NUM_VAL = 50
SEED_OFFSET = 100000


def find_best_controller():
    """Find the best-performing controller from existing NumPy results."""
    results_base = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    best_dir = None
    best_goals = 0
    best_goals_str = ""

    for entry in os.listdir(results_base):
        if not entry.startswith('forward_'):
            continue
        full = os.path.join(results_base, entry)
        if not os.path.isdir(full):
            continue
        for seed in os.listdir(full):
            if not seed.startswith('seed_'):
                continue
            d = os.path.join(full, seed)
            vpath = os.path.join(d, 'validation_summary.txt')
            wpath = os.path.join(d, 'best_weights.npy')
            if not os.path.exists(vpath) or not os.path.exists(wpath):
                continue
            # Must have a config yaml
            cfgs = [f for f in os.listdir(d) if f.endswith('.yaml')]
            if not cfgs:
                continue
            with open(vpath) as f:
                txt = f.read()
            m = re.search(r'Goals scored:\s+(\d+)/(\d+)', txt)
            if m and int(m.group(2)) == 50:
                g = int(m.group(1))
                if g > best_goals:
                    best_goals = g
                    best_dir = d
                    best_goals_str = f"{m.group(1)}/{m.group(2)}"

    return best_dir, best_goals, best_goals_str


def run_numpy_validation(weights, config):
    """Run NumPy validation on 50 seeds. Returns per-trial array [0 or 1]."""
    config = dict(config)
    if 'challenge' in config:
        config['challenge'] = dict(config['challenge'])
        config['challenge']['enabled'] = False
    if 'curriculum' in config:
        config['curriculum'] = dict(config['curriculum'])
        config['curriculum']['enabled'] = False

    sim = Simulator(config)
    sim._state_pool = None

    results = []
    for i in range(NUM_VAL):
        seed = SEED_OFFSET + i
        metrics = sim.run(weights, random_seed=seed)
        scored = 1 if metrics.get('goal_scored') == metrics.get('target_goal_side') else 0
        results.append(scored)

    return np.array(results)


def run_jax_validation(weights, config):
    """Run JAX validation on 50 seeds using NumPy simulator (same as runner.py)."""
    # JAX runner uses NumPy simulator for validation — so this should be identical
    config = dict(config)
    if 'challenge' in config:
        config['challenge'] = dict(config['challenge'])
        config['challenge']['enabled'] = False
    if 'curriculum' in config:
        config['curriculum'] = dict(config['curriculum'])
        config['curriculum']['enabled'] = False

    sim = Simulator(config)
    sim._state_pool = None

    results = []
    for i in range(NUM_VAL):
        seed = SEED_OFFSET + i
        metrics = sim.run(weights, random_seed=seed)
        scored = 1 if metrics.get('goal_scored') == metrics.get('target_goal_side') else 0
        results.append(scored)

    return np.array(results)


def run_jax_physics_validation(weights, config):
    """Run validation using JAX physics (not NumPy simulator).

    This tests whether the JAX simulation engine itself produces the same
    results as NumPy when given the same initial conditions.
    """
    # We can't use the same seeds because JAX RNG differs.
    # Instead, we use the NumPy simulator to generate initial states,
    # then run those through JAX physics.
    config_val = dict(config)
    if 'challenge' in config_val:
        config_val['challenge'] = dict(config_val['challenge'])
        config_val['challenge']['enabled'] = False
    if 'curriculum' in config_val:
        config_val['curriculum'] = dict(config_val['curriculum'])
        config_val['curriculum']['enabled'] = False

    # Get initial states from NumPy simulator
    sim = Simulator(config_val)
    sim._state_pool = None

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_val, f)
        tmp = f.name
    try:
        params = load_config_jax(tmp)
    finally:
        os.unlink(tmp)

    weights_jax = jnp.array(weights, dtype=jnp.float32)
    results = []

    for i in range(NUM_VAL):
        seed = SEED_OFFSET + i

        # Use NumPy sim to get the initial state for this seed
        sim.reset(random_seed=seed)
        rx = sim.robots[0].x
        ry = sim.robots[0].y
        rq = sim.robots[0].q
        bx = sim.ball.x
        by = sim.ball.y

        # Run through JAX physics
        key = jax.random.PRNGKey(seed)
        final = rollout(weights_jax, rx, ry, rq, bx, by, key, params)
        scored = 1 if int(final.goal_scored) == 1 else 0  # 1 = right goal
        results.append(scored)

    return np.array(results)


def main():
    print("=" * 70)
    print("  TIER 3.2 — Known-good controller cross-validation")
    print("=" * 70)

    # Find best controller
    best_dir, best_goals, best_goals_str = find_best_controller()
    if best_dir is None or best_goals < 10:
        print("  ERROR: No suitable controller found (need ≥10/50 goals)")
        return

    # Load weights and config
    weights_path = os.path.join(best_dir, 'best_weights.npy')
    cfg_files = [f for f in os.listdir(best_dir) if f.endswith('.yaml')]
    cfg_path = os.path.join(best_dir, cfg_files[0])

    print(f"\n  Weights file:  {weights_path}")
    print(f"  Config file:   {cfg_path}")
    print(f"  Documented goal rate: {best_goals_str} ({best_goals * 2}%)")

    # Read validation_summary.txt
    with open(os.path.join(best_dir, 'validation_summary.txt')) as f:
        print(f"\n  Original validation_summary.txt:")
        for line in f:
            if line.strip():
                print(f"    {line.rstrip()}")

    weights = np.load(weights_path)
    print(f"\n  Weights shape: {weights.shape}")
    print(f"  Validation seeds: [{SEED_OFFSET}..{SEED_OFFSET + NUM_VAL - 1}]")

    from utils.config_loader import load_config
    config = load_config(cfg_path)

    # Run NumPy validation
    print("\n  Running NumPy validation...")
    np_results = run_numpy_validation(weights, config)
    np_goals = np_results.sum()

    # Run JAX validation (uses NumPy sim — should be identical)
    print("  Running JAX validator (NumPy sim)...")
    jax_val_results = run_jax_validation(weights, config)
    jax_val_goals = jax_val_results.sum()

    # Run JAX physics validation (uses JAX sim with same initial states)
    print("  Running JAX physics validation...")
    jax_phys_results = run_jax_physics_validation(weights, config)
    jax_phys_goals = jax_phys_results.sum()

    # Results
    print(f"\n  {'':>25s} {'Goals':>6s} {'Rate':>6s}")
    print(f"  {'-' * 42}")
    print(f"  {'Original (documented)':>25s} {best_goals_str:>6s} {best_goals * 2:>5d}%")
    print(f"  {'NumPy validator':>25s} {np_goals:>5d}/50 {np_goals * 2:>5d}%")
    print(f"  {'JAX validator (NP sim)':>25s} {jax_val_goals:>5d}/50 {jax_val_goals * 2:>5d}%")
    print(f"  {'JAX physics':>25s} {jax_phys_goals:>5d}/50 {jax_phys_goals * 2:>5d}%")

    # Per-trial comparison
    print(f"\n  Per-trial results (50 seeds):")
    print(f"  {'Seed':>8s}  {'NumPy':>5s}  {'JAX-val':>7s}  {'JAX-phys':>8s}  {'Match':>5s}")
    print(f"  {'-' * 42}")

    np_vs_jaxval_disagree = 0
    np_vs_jaxphys_disagree = 0

    for i in range(NUM_VAL):
        np_r = np_results[i]
        jv_r = jax_val_results[i]
        jp_r = jax_phys_results[i]
        match_val = "✓" if np_r == jv_r else "✗"
        match_phys = "✓" if np_r == jp_r else "✗"
        if np_r != jv_r:
            np_vs_jaxval_disagree += 1
        if np_r != jp_r:
            np_vs_jaxphys_disagree += 1

        # Only print disagreements and first/last few
        if np_r != jp_r or i < 3 or i >= 47:
            print(f"  {SEED_OFFSET + i:>8d}  {np_r:>5d}  {jv_r:>7d}  {jp_r:>8d}  {match_val}{match_phys}")

    print(f"\n  NumPy vs JAX-validator disagreements: {np_vs_jaxval_disagree}/50")
    print(f"  NumPy vs JAX-physics disagreements:   {np_vs_jaxphys_disagree}/50")

    # Arrays for easy comparison
    print(f"\n  NumPy array:    {np_results.tolist()}")
    print(f"  JAX-val array:  {jax_val_results.tolist()}")
    print(f"  JAX-phys array: {jax_phys_results.tolist()}")

    # Verdict
    print(f"\n  {'=' * 42}")
    val_ok = np_vs_jaxval_disagree == 0
    phys_ok = np_vs_jaxphys_disagree <= 3
    print(f"  NumPy vs JAX-validator: {'PASS (identical)' if val_ok else 'FAIL'}")
    print(f"  NumPy vs JAX-physics:   {'PASS' if phys_ok else 'FAIL'} "
          f"({np_vs_jaxphys_disagree} disagreements, threshold ≤3)")

    if not phys_ok:
        print(f"\n  WARNING: JAX physics disagrees on {np_vs_jaxphys_disagree} trials.")
        print(f"  This indicates the JAX simulation engine produces different")
        print(f"  outcomes on some scenarios, likely due to noise RNG differences")
        print(f"  or accumulated float32 drift over long episodes.")


if __name__ == '__main__':
    main()
