"""Noise debugging tests — find the source of distributional differences.

Tests:
1. AR(1) stationarity: init std and post-update std should both be σ
2. AR(1) long-run variance: run 500 steps, check variance stays at σ²
3. Bias isolation: check bias is sampled once per trial, not re-sampled
4. Full sensor noise pipeline: compare final gain distribution

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_noise_debug.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import jax
import jax.numpy as jnp
import numpy as np


def test_0_micro_test():
    """The exact micro-test from the specification."""
    print("=== Test 0: Exact micro-test from spec ===")

    sigma = 0.04
    rho = 0.92
    n = 10000

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n)
    init_states = jax.vmap(lambda k: jax.random.normal(k) * sigma)(keys)
    print(f"  Init std: {float(init_states.std()):.5f}  (should be {sigma})")

    update_keys = jax.random.split(jax.random.PRNGKey(1), n)
    def one_update(state, k):
        inn_std = sigma * jnp.sqrt(1 - rho**2)
        return rho * state + jax.random.normal(k) * inn_std
    updated = jax.vmap(one_update)(init_states, update_keys)
    updated_std = float(updated.std())
    print(f"  After 1 update std: {updated_std:.5f}  (should still be {sigma})")

    ok = abs(updated_std - sigma) < 0.005
    if not ok:
        ratio = updated_std / sigma
        print(f"  RATIO: {ratio:.3f} — if ~1.36, variance is doubling")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_0b_init_noise_ar_std():
    """Scenario A: Check sensor_ar.std() right after init_noise."""
    print("\n=== Test 0b: sensor_ar.std() after init_noise ===")

    from jax_sim.noise import init_noise
    from jax_sim.tests.test_components import _make_params

    params = _make_params()._replace(sensor_noise_enabled=True)

    # Run init_noise 5000 times, collect sensor_ar values
    all_ar = []
    for i in range(5000):
        key = jax.random.PRNGKey(i)
        noise = init_noise(key, params)
        all_ar.append(np.array(noise.sensor_ar))

    all_ar = np.array(all_ar)  # (5000, 8)
    per_channel_std = all_ar.std(axis=0)
    overall_std = all_ar.std()

    print(f"  Per-channel std: {per_channel_std}")
    print(f"  Overall std:     {overall_std:.5f}")
    print(f"  Expected:        {params.sensor_relative_std:.5f}")

    ok = abs(overall_std - params.sensor_relative_std) < 0.005
    if not ok:
        print(f"  WRONG! Got {overall_std:.5f}, expected ~{params.sensor_relative_std}")
        print(f"  If ~{np.sqrt(params.sensor_relative_std**2 + params.sensor_bias_std**2):.5f}, bias is leaking into AR init")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_ar1_stationarity():
    """Scenario A+B: Check AR(1) init and update produce correct variance."""
    print("\n=== Test 1: AR(1) stationarity (via update_ar1) ===")

    sigma = 0.04
    rho = 0.92
    n = 10000

    key = jax.random.PRNGKey(0)

    # Sample initial states from N(0, σ²)
    keys = jax.random.split(key, n)
    init_states = jax.vmap(lambda k: jax.random.normal(k) * sigma)(keys)
    init_std = float(init_states.std())
    print(f"  Init std:          {init_std:.5f}  (expected: {sigma:.5f})")

    # Update each once using the JAX update_ar1 formula
    from jax_sim.noise import update_ar1

    update_keys = jax.random.split(jax.random.PRNGKey(1), n)

    def one_update(state, k):
        # update_ar1 expects arrays, wrap scalar
        s = jnp.array([state])
        target = jnp.array([sigma])
        result = update_ar1(s, k, target, rho)
        return result[0]

    updated = jax.vmap(one_update)(init_states, update_keys)
    updated_std = float(updated.std())
    print(f"  After 1 update:    {updated_std:.5f}  (expected: {sigma:.5f})")

    # Run 100 updates to check long-run stability
    def run_chain(key_init):
        k1, k2 = jax.random.split(key_init)
        state = jax.random.normal(k1) * sigma

        def step(carry, k):
            s = jnp.array([carry])
            s_new = update_ar1(s, k, jnp.array([sigma]), rho)
            return s_new[0], s_new[0]

        step_keys = jax.random.split(k2, 100)
        final, trajectory = jax.lax.scan(step, state, step_keys)
        return trajectory

    chain_keys = jax.random.split(jax.random.PRNGKey(2), 1000)
    all_chains = jax.vmap(run_chain)(chain_keys)  # (1000, 100)

    # Variance at each timestep (across 1000 chains)
    per_step_std = all_chains.std(axis=0)
    mean_std = float(per_step_std.mean())
    min_std = float(per_step_std.min())
    max_std = float(per_step_std.max())
    print(f"  Long-run std (100 steps, 1000 chains):")
    print(f"    mean={mean_std:.5f}  min={min_std:.5f}  max={max_std:.5f}")
    print(f"    expected: all ≈ {sigma:.5f}")

    init_ok = abs(init_std - sigma) < 0.005
    update_ok = abs(updated_std - sigma) < 0.005
    longrun_ok = abs(mean_std - sigma) < 0.005
    ok = init_ok and update_ok and longrun_ok
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not init_ok:
            print(f"    → Init std wrong: {init_std:.5f} vs expected {sigma}")
        if not update_ok:
            print(f"    → Post-update std wrong: {updated_std:.5f} vs expected {sigma}")
            ratio = updated_std / sigma
            print(f"    → Ratio: {ratio:.3f} (if ~1.36, variance is doubling)")
        if not longrun_ok:
            print(f"    → Long-run std drifting: {mean_std:.5f}")
    return ok


def test_bias_not_resampled():
    """Scenario C: Check bias is fixed per trial, not re-sampled on each call."""
    print("\n=== Test 2: Bias stability per trial ===")

    from jax_sim.noise import init_noise, apply_sensor_noise
    from jax_sim.tests.test_components import _make_params

    params = _make_params()._replace(sensor_noise_enabled=True)

    key = jax.random.PRNGKey(42)
    noise = init_noise(key, params)

    bias_before = np.array(noise.sensor_bias)

    # Apply noise 10 times
    raw_distances = jnp.ones(8) * 0.15
    for i in range(10):
        k = jax.random.PRNGKey(100 + i)
        _, noise = apply_sensor_noise(raw_distances, noise, k, params)

    bias_after = np.array(noise.sensor_bias)

    bias_diff = np.max(np.abs(bias_before - bias_after))
    print(f"  Bias before: {bias_before[:4]}")
    print(f"  Bias after:  {bias_after[:4]}")
    print(f"  Max diff: {bias_diff:.2e}")
    ok = bias_diff < 1e-10
    print(f"  Bias unchanged: {'YES' if ok else 'NO — BUG!'}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_sensor_gain_distribution():
    """Full pipeline: check the effective gain (1 + bias + ar) distribution."""
    print("\n=== Test 3: Sensor gain distribution ===")

    from jax_sim.noise import init_noise, apply_sensor_noise
    from jax_sim.tests.test_components import _make_params

    params = _make_params()._replace(sensor_noise_enabled=True)

    n_trials = 2000
    fwd_idx = 4  # forward sensor
    raw_distances = jnp.ones(8) * 0.15  # 15cm

    gains = []
    for i in range(n_trials):
        key = jax.random.PRNGKey(i)
        k1, k2 = jax.random.split(key)
        noise = init_noise(k1, params)
        noisy, _ = apply_sensor_noise(raw_distances, noise, k2, params)
        # gain = noisy / raw
        gain = float(noisy[fwd_idx]) / 0.15
        gains.append(gain)

    gains = np.array(gains)

    # Expected: gain = 1 + bias + ar
    # bias ~ N(0, 0.015²), ar ~ N(0, 0.04²)
    # Total noise std: sqrt(0.015² + 0.04²) = sqrt(0.000225 + 0.0016) = 0.0427
    # So gain ~ N(1, 0.0427²)
    expected_mean = 1.0
    expected_std = np.sqrt(params.sensor_bias_std**2 + params.sensor_relative_std**2)

    print(f"  Gain mean: {gains.mean():.5f}  (expected: {expected_mean:.5f})")
    print(f"  Gain std:  {gains.std():.5f}  (expected: {expected_std:.5f})")
    print(f"  Gain min:  {gains.min():.5f}")
    print(f"  Gain max:  {gains.max():.5f}")

    mean_ok = abs(gains.mean() - expected_mean) < 0.005
    std_ok = abs(gains.std() - expected_std) / expected_std < 0.15  # within 15%
    ok = mean_ok and std_ok
    print(f"  Mean within 0.005: {'YES' if mean_ok else 'NO'}")
    print(f"  Std within 15%:    {'YES' if std_ok else 'NO'}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_ar1_vs_numpy():
    """Direct comparison: run NumPy AR(1) vs JAX AR(1) and compare distributions."""
    print("\n=== Test 4: AR(1) NumPy vs JAX distribution ===")

    sigma = 0.04
    rho = 0.92
    n_trials = 2000
    n_steps = 50

    # NumPy AR(1)
    np_rng = np.random.RandomState(42)
    np_final_states = []
    for _ in range(n_trials):
        state = np_rng.normal(0, sigma)  # stationary init
        for _ in range(n_steps):
            inn_std = sigma * np.sqrt(1 - rho**2)
            state = rho * state + np_rng.normal(0, inn_std)
        np_final_states.append(state)
    np_arr = np.array(np_final_states)

    # JAX AR(1)
    from jax_sim.noise import update_ar1

    def run_jax_chain(key):
        k1, k2 = jax.random.split(key)
        state = jax.random.normal(k1, (1,)) * sigma
        def step(s, k):
            s_new = update_ar1(s, k, jnp.array([sigma]), rho)
            return s_new, None
        keys = jax.random.split(k2, n_steps)
        final, _ = jax.lax.scan(step, state, keys)
        return final[0]

    jax_keys = jax.random.split(jax.random.PRNGKey(42), n_trials)
    jax_arr = np.array(jax.vmap(run_jax_chain)(jax_keys))

    print(f"  {'':>10s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'NumPy':>10s} {np_arr.mean():10.6f} {np_arr.std():10.6f} {np_arr.min():10.6f} {np_arr.max():10.6f}")
    print(f"  {'JAX':>10s} {jax_arr.mean():10.6f} {jax_arr.std():10.6f} {jax_arr.min():10.6f} {jax_arr.max():10.6f}")

    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(np_arr, jax_arr)
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")
    print(f"  Equivalent (p > 0.01): {'YES' if ks_p > 0.01 else 'NO'}")

    mean_ok = abs(np_arr.mean() - jax_arr.mean()) < 0.003
    std_ok = abs(np_arr.std() - jax_arr.std()) / np_arr.std() < 0.10
    ok = mean_ok and std_ok
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_full_ir_pipeline_comparison():
    """Compare full IR pipeline output distributions between NumPy and JAX."""
    print("\n=== Test 5: Full IR pipeline distribution comparison ===")
    print("  Robot at (0,0,0), ball at (0.10, 0), sensor noise ON")
    print("  Forward sensor (idx 4), 2000 trials")

    import yaml, tempfile
    from core.simulator import Simulator
    from jax_sim.config import load_config as load_config_jax
    from jax_sim.noise import init_noise, apply_sensor_noise
    from jax_sim.sensors import get_ir_readings, _normalize_sensors
    from jax_sim.types import RobotState, BallState

    config = {
        'robot': {'wheel_radius': 0.021, 'half_wheelbase': 0.0527,
                  'body_radius': 0.0704, 'mass': 0.566,
                  'max_wheel_speed': 14.3, 'max_wheel_cmd_delta': 0.15,
                  'wall_collision': 'push'},
        'pitch': {'length': 2.8, 'width': 2.44, 'goal_width': 0.75,
                  'goal_depth': 0.19, 'goal_area_width': 1.03,
                  'goal_area_depth': 0.19, 'penalty_area_width': 1.5,
                  'penalty_area_depth': 0.47, 'penalty_spot_distance': 0.38,
                  'center_circle_radius': 0.28, 'corner_arc_radius': 0.09,
                  'wall_thickness': 0.04, 'wall_mode': 'open'},
        'ball': {'radius': 0.0335, 'mass': 0.0577, 'gravity': 9.81,
                 'rolling_friction': 0.04, 'wall_restitution': 0.745,
                 'inertia_factor': 0.6667, 'sliding_friction': 0.23,
                 'wall_friction': 0.23, 'robot_friction': 0.23,
                 'robot_restitution': 0.5, 'spin_damping': 0.0,
                 'placement': 'random',
                 'x_range': [-1.3665, 1.3665], 'y_range': [-1.1865, 1.1865]},
        'teams': {'blue': {'num_robots': 1, 'placement': 'random',
                           'x_range': [-1.3296, 1.3296],
                           'y_range': [-1.1496, 1.1496],
                           'q_range': [-3.14159, 3.14159]},
                  'yellow': {'num_robots': 0}},
        'sensors': {'angles': [-2.53, -1.571, -0.785, -0.175, 0.175, 0.785, 1.571, 2.53],
                    'max_range': 0.25, 'min_range': 0.005,
                    'noise': {'enabled': True, 'model': 'correlated',
                              'relative_std': 0.04, 'bias_std': 0.015, 'rho': 0.92},
                    'normalization': 'minmax'},
        'vision': {'mode': 'frontal', 'horizontal_fov_deg': 131.0,
                   'ball_max_range': 1.9, 'goal_max_range': 3.6,
                   'noise': {'enabled': False}},
        'simulation': {'global_seed': 0, 'max_steps': 500, 'timestep': 0.1,
                       'physics_timestep': 0.01, 'num_trials': 1,
                       'fitness_function': 'penalty_sparse', 'trajectory_stride': 1},
        'neural_network': {'type': 'elman', 'hidden_size': 5, 'output_size': 2,
                           'vision_inputs': 4, 'activation': 'sigmoid',
                           'wheel_output_mapping': 'floreano'},
        'curriculum': {'enabled': False},
        'challenge': {'enabled': False},
        'motor': {'noise': {'enabled': False}},
        'fitness_params': {'goal_reward': 1.0, 'time_bonus': False},
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        params = load_config_jax(tmp)
    finally:
        os.unlink(tmp)

    hl = config['pitch']['length'] / 2.0
    n_trials = 2000
    fwd_idx = 4

    # NumPy
    np_readings = []
    sim = Simulator(config)
    for i in range(n_trials):
        sim.reset(random_seed=i)
        sim.robots[0].x, sim.robots[0].y, sim.robots[0].q = 0.0, 0.0, 0.0
        sim.ball.x, sim.ball.y = 0.10, 0.0
        inputs = sim.robots[0].get_all_inputs([], sim.ball, hl)
        np_readings.append(inputs[fwd_idx])
    np_arr = np.array(np_readings)

    # JAX
    jax_readings = []
    jax_robot = RobotState(x=jnp.float32(0.0), y=jnp.float32(0.0),
                            q=jnp.float32(0.0),
                            left_actual=jnp.float32(0.0),
                            right_actual=jnp.float32(0.0))
    jax_ball = BallState(x=jnp.float32(0.10), y=jnp.float32(0.0),
                          vx=jnp.float32(0.0), vy=jnp.float32(0.0),
                          omega=jnp.float32(0.0))

    for i in range(n_trials):
        key = jax.random.PRNGKey(i)
        k1, k2 = jax.random.split(key)
        noise = init_noise(k1, params)
        raw = get_ir_readings(jax_robot, jax_ball, params)
        noisy, _ = apply_sensor_noise(raw, noise, k2, params)
        clipped = jnp.clip(noisy, params.sensor_min, params.sensor_max)
        normalized = _normalize_sensors(clipped, params)
        jax_readings.append(float(normalized[fwd_idx]))
    jax_arr = np.array(jax_readings)

    se = np_arr.std() / np.sqrt(n_trials)

    print(f"\n  {'':>10s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'NumPy':>10s} {np_arr.mean():10.6f} {np_arr.std():10.6f} {np_arr.min():10.6f} {np_arr.max():10.6f}")
    print(f"  {'JAX':>10s} {jax_arr.mean():10.6f} {jax_arr.std():10.6f} {jax_arr.min():10.6f} {jax_arr.max():10.6f}")
    print(f"  Std error: {se:.6f}")
    print(f"  Mean diff / SE: {abs(np_arr.mean() - jax_arr.mean()) / max(se, 1e-10):.2f}")

    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(np_arr, jax_arr)
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")
    print(f"  Equivalent (p > 0.01): {'YES' if ks_p > 0.01 else 'NO'}")

    mean_ok = abs(np_arr.mean() - jax_arr.mean()) < 2 * se
    std_ok = abs(np_arr.std() - jax_arr.std()) / max(np_arr.std(), 1e-10) < 0.15
    ok = mean_ok and std_ok
    print(f"  Mean within 2 SE: {'YES' if mean_ok else 'NO'}")
    print(f"  Std within 15%:   {'YES' if std_ok else 'NO'}")
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Save histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(np_arr.min(), jax_arr.min()),
                           max(np_arr.max(), jax_arr.max()), 60)
        ax.hist(np_arr, bins=bins, alpha=0.5, density=True,
                label=f'NumPy (μ={np_arr.mean():.4f}, σ={np_arr.std():.4f})')
        ax.hist(jax_arr, bins=bins, alpha=0.5, density=True,
                label=f'JAX (μ={jax_arr.mean():.4f}, σ={jax_arr.std():.4f})')
        ax.set_xlabel('Normalized IR reading (forward sensor, ball at 10cm)')
        ax.set_ylabel('Density')
        ax.set_title(f'Full IR Pipeline: NumPy vs JAX (n={n_trials}, KS p={ks_p:.3f})')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = 'jax_sim/tests/noise_debug_ir_pipeline.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {save_path}")
    except ImportError:
        pass

    return ok


def main():
    print("=" * 60)
    print("  NOISE DEBUGGING TESTS")
    print("=" * 60)
    print()

    results = []
    results.append(("0. Exact micro-test", test_0_micro_test()))
    results.append(("0b. sensor_ar.std() after init", test_0b_init_noise_ar_std()))
    results.append(("1. AR(1) stationarity via update_ar1", test_ar1_stationarity()))
    results.append(("2. Bias not re-sampled", test_bias_not_resampled()))
    results.append(("3. Sensor gain distribution", test_sensor_gain_distribution()))
    results.append(("4. AR(1) NumPy vs JAX", test_ar1_vs_numpy()))
    results.append(("5. Full IR pipeline comparison", test_full_ir_pipeline_comparison()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {sum(1 for _, ok in results if ok)}/{len(results)} passed")


if __name__ == '__main__':
    main()
