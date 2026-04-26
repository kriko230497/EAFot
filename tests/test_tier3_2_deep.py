"""Tier 3.2 deep investigation — are disagreements noise or bias?

Test 1: Run JAX physics 5 times with different RNG seeds. If goal rate
        swings (27-31), it's noise. If stable (28-29), it's bias.

Test 2: Run both with ALL noise disabled. If still disagreements,
        it's physics drift. If 0 disagreements, it's 100% RNG.

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_tier3_2_deep.py
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


NUM_VAL = 50
SEED_OFFSET = 100000


def find_best_controller():
    """Find the best-performing controller from existing NumPy results."""
    results_base = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    best_dir = None
    best_goals = 0

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
            cfgs = [f for f in os.listdir(d) if f.endswith('.yaml')]
            if not cfgs:
                continue
            with open(vpath) as f:
                txt = f.read()
            m = re.search(r'Goals scored:\s+(\d+)/50', txt)
            if m:
                g = int(m.group(1))
                if g > best_goals:
                    best_goals = g
                    best_dir = d

    return best_dir, best_goals


def get_numpy_baseline(weights, config):
    """NumPy validation — the ground truth."""
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
        metrics = sim.run(weights, random_seed=SEED_OFFSET + i)
        scored = 1 if metrics.get('goal_scored') == metrics.get('target_goal_side') else 0
        results.append(scored)
    return np.array(results)


def get_initial_states(config):
    """Get the 50 initial states from NumPy sim (deterministic)."""
    config = dict(config)
    if 'challenge' in config:
        config['challenge'] = dict(config['challenge'])
        config['challenge']['enabled'] = False
    if 'curriculum' in config:
        config['curriculum'] = dict(config['curriculum'])
        config['curriculum']['enabled'] = False

    sim = Simulator(config)
    sim._state_pool = None

    states = []
    for i in range(NUM_VAL):
        sim.reset(random_seed=SEED_OFFSET + i)
        states.append((
            sim.robots[0].x, sim.robots[0].y, sim.robots[0].q,
            sim.ball.x, sim.ball.y,
        ))
    return states


def run_jax_physics(weights, config, states, rng_offset=0, use_x64=False):
    """Run JAX physics on given initial states with an RNG offset."""
    config_jax = dict(config)
    config_jax['jax'] = {'use_x64': use_x64}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_jax, f)
        tmp = f.name
    try:
        params = load_config_jax(tmp)
    finally:
        os.unlink(tmp)

    dtype = jnp.float64 if use_x64 else jnp.float32
    weights_jax = jnp.array(weights, dtype=dtype)
    results = []
    for i in range(NUM_VAL):
        rx, ry, rq, bx, by = states[i]
        key = jax.random.PRNGKey(SEED_OFFSET + i + rng_offset)
        final = rollout(weights_jax, rx, ry, rq, bx, by, key, params)
        scored = 1 if int(final.goal_scored) == 1 else 0
        results.append(scored)
    return np.array(results)


def main():
    print("=" * 70)
    print("  TIER 3.2 DEEP — Noise vs Bias investigation")
    print("=" * 70)

    best_dir, best_goals = find_best_controller()
    if best_dir is None or best_goals < 10:
        print("  ERROR: No suitable controller found")
        return

    weights = np.load(os.path.join(best_dir, 'best_weights.npy'))
    cfg_files = [f for f in os.listdir(best_dir) if f.endswith('.yaml')]
    cfg_path = os.path.join(best_dir, cfg_files[0])

    from utils.config_loader import load_config
    config = load_config(cfg_path)

    print(f"  Controller: {best_dir}")
    print(f"  Documented: {best_goals}/50")

    # Get NumPy baseline
    print("\n  Running NumPy baseline...")
    np_results = get_numpy_baseline(weights, config)
    np_goals = np_results.sum()
    print(f"  NumPy: {np_goals}/50")

    # Get initial states (shared between all JAX runs)
    states = get_initial_states(config)

    # ═══════════════════════════════════════════════════
    # TEST 1: Multiple JAX RNG seeds — noise or bias?
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: Is disagreement noise or bias?")
    print("  Running JAX physics 5 times with different RNG offsets")
    print("=" * 70)

    jax_runs = []
    jax_goal_counts = []
    for run_idx in range(5):
        rng_offset = run_idx * 1000
        jax_res = run_jax_physics(weights, config, states, rng_offset)
        jax_runs.append(jax_res)
        jax_goal_counts.append(jax_res.sum())
        disagree = (np_results != jax_res).sum()
        print(f"  Run {run_idx} (offset={rng_offset:>5d}): "
              f"{jax_res.sum():>2d}/50 goals, "
              f"{disagree} disagreements with NumPy")

    jax_goal_arr = np.array(jax_goal_counts)
    print(f"\n  JAX goal counts: {jax_goal_counts}")
    print(f"  Mean: {jax_goal_arr.mean():.1f}, Std: {jax_goal_arr.std():.1f}, "
          f"Range: [{jax_goal_arr.min()}, {jax_goal_arr.max()}]")

    if jax_goal_arr.std() >= 1.5:
        print("  → NOISE: goal rate swings across runs — disagreements are stochastic")
    else:
        print("  → POSSIBLE BIAS: goal rate stable across runs — investigate further")

    # Check which trials are consistently different
    print(f"\n  Per-trial consistency (across 5 JAX runs):")
    print(f"  {'Trial':>7s}  {'NumPy':>5s}  {'JAX votes':>9s}  {'Verdict':>10s}")
    print(f"  {'-' * 40}")

    always_disagree = 0
    sometimes_disagree = 0
    for i in range(NUM_VAL):
        jax_votes = sum(r[i] for r in jax_runs)
        if np_results[i] == 1 and jax_votes == 0:
            verdict = "ALWAYS OFF"
            always_disagree += 1
            print(f"  {SEED_OFFSET+i:>7d}  {np_results[i]:>5d}  {jax_votes:>5d}/5    {verdict}")
        elif np_results[i] == 0 and jax_votes == 5:
            verdict = "ALWAYS ON"
            always_disagree += 1
            print(f"  {SEED_OFFSET+i:>7d}  {np_results[i]:>5d}  {jax_votes:>5d}/5    {verdict}")
        elif np_results[i] != (1 if jax_votes >= 3 else 0):
            verdict = "MARGINAL"
            sometimes_disagree += 1
            print(f"  {SEED_OFFSET+i:>7d}  {np_results[i]:>5d}  {jax_votes:>5d}/5    {verdict}")

    print(f"\n  Always disagree: {always_disagree} trials (bias)")
    print(f"  Sometimes disagree: {sometimes_disagree} trials (noise)")

    # ═══════════════════════════════════════════════════
    # TEST 2: No noise — physics only
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: All noise disabled — pure physics comparison")
    print("=" * 70)

    # Disable all noise in config
    config_no_noise = dict(config)
    if 'sensors' in config_no_noise:
        config_no_noise['sensors'] = dict(config_no_noise['sensors'])
        config_no_noise['sensors']['noise'] = {'enabled': False}
    if 'vision' in config_no_noise:
        config_no_noise['vision'] = dict(config_no_noise['vision'])
        config_no_noise['vision']['noise'] = {'enabled': False}
    if 'motor' in config_no_noise:
        config_no_noise['motor'] = dict(config_no_noise['motor'])
        config_no_noise['motor']['noise'] = {'enabled': False}

    # NumPy no-noise
    print("  Running NumPy (no noise)...")
    np_nn = get_numpy_baseline(weights, config_no_noise)
    np_nn_goals = np_nn.sum()

    # Get initial states from no-noise config
    states_nn = get_initial_states(config_no_noise)

    # JAX no-noise
    print("  Running JAX physics (no noise)...")
    jax_nn = run_jax_physics(weights, config_no_noise, states_nn, rng_offset=0)
    jax_nn_goals = jax_nn.sum()

    disagree_nn = (np_nn != jax_nn).sum()

    print(f"\n  NumPy (no noise): {np_nn_goals}/50")
    print(f"  JAX (no noise):   {jax_nn_goals}/50")
    print(f"  Disagreements:    {disagree_nn}/50")

    if disagree_nn == 0:
        print("  → CONCLUSION: 0 disagreements without noise.")
        print("     All disagreements in Test 1 are 100% due to RNG differences.")
        print("     The physics engines are equivalent.")
    elif disagree_nn <= 2:
        print(f"  → MINOR: {disagree_nn} disagreements — likely float32 precision drift")
        print("     on marginal scenarios (ball barely crosses goal line).")
    else:
        print(f"  → CONCERN: {disagree_nn} disagreements even without noise.")
        print("     This indicates a real physics difference.")

    # Show the no-noise arrays
    print(f"\n  NumPy (no noise): {np_nn.tolist()}")
    print(f"  JAX (no noise):   {jax_nn.tolist()}")

    # Show which trials disagree
    if disagree_nn > 0:
        print(f"\n  Disagreeing trials (no noise, float32):")
        for i in range(NUM_VAL):
            if np_nn[i] != jax_nn[i]:
                rx, ry, rq, bx, by = states_nn[i]
                print(f"    Seed {SEED_OFFSET+i}: NumPy={np_nn[i]} JAX={jax_nn[i]}  "
                      f"robot=({rx:.3f},{ry:.3f},{rq:.3f}) ball=({bx:.3f},{by:.3f})")

    # ═══════════════════════════════════════════════════
    # TEST 3: No noise + float64 — eliminate precision as factor
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 3: No noise + float64 (jax.use_x64=True)")
    print("=" * 70)

    print("  Running JAX physics (no noise, x64)...")
    jax_nn_x64 = run_jax_physics(weights, config_no_noise, states_nn,
                                  rng_offset=0, use_x64=True)
    jax_nn_x64_goals = jax_nn_x64.sum()
    disagree_x64 = (np_nn != jax_nn_x64).sum()

    print(f"\n  NumPy (no noise, f64):     {np_nn_goals}/50")
    print(f"  JAX (no noise, f32):       {jax_nn_goals}/50  ({disagree_nn} disagreements)")
    print(f"  JAX (no noise, f64):       {jax_nn_x64_goals}/50  ({disagree_x64} disagreements)")

    if disagree_x64 == 0:
        print("  → float64 eliminates ALL disagreements.")
        print("     The physics code is identical — float32 precision was the only difference.")
    elif disagree_x64 < disagree_nn:
        print(f"  → float64 reduces disagreements from {disagree_nn} to {disagree_x64}.")
        print("     Some drift is precision, some may be code difference.")
    else:
        print(f"  → float64 did NOT help ({disagree_x64} vs {disagree_nn}).")
        print("     The disagreements are from a code difference, not precision.")

    if disagree_x64 > 0:
        print(f"\n  Disagreeing trials (no noise, x64):")
        for i in range(NUM_VAL):
            if np_nn[i] != jax_nn_x64[i]:
                rx, ry, rq, bx, by = states_nn[i]
                print(f"    Seed {SEED_OFFSET+i}: NumPy={np_nn[i]} JAX_x64={jax_nn_x64[i]}  "
                      f"robot=({rx:.3f},{ry:.3f},{rq:.3f}) ball=({bx:.3f},{by:.3f})")

    print(f"\n  NumPy array (no noise):     {np_nn.tolist()}")
    print(f"  JAX f32 array (no noise):   {jax_nn.tolist()}")
    print(f"  JAX f64 array (no noise):   {jax_nn_x64.tolist()}")


if __name__ == '__main__':
    main()
