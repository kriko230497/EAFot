"""Tier 2 — Noise distribution tests.

2.1: Sensor noise distribution (1000 trials, forward IR sensor)
2.2: Vision noise close-range degradation
2.3: Vision dropout rate

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_tier2_noise.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import yaml
import tempfile

from core.simulator import Simulator
from core.robot import Robot
from core.ball import Ball
from core.pitch import Pitch

from jax_sim.config import load_config as load_config_jax
from jax_sim.types import RobotState, BallState
from jax_sim.sensors import get_ir_readings, _normalize_sensors
from jax_sim.noise import init_noise, apply_sensor_noise
from jax_sim.simulator import init_state, step_fn
from jax_sim.neural_network import num_weights


def _make_config(sensor_noise=True, vision_noise=True):
    return {
        'robot': {
            'wheel_radius': 0.021, 'half_wheelbase': 0.0527,
            'body_radius': 0.0704, 'mass': 0.566,
            'max_wheel_speed': 14.3, 'max_wheel_cmd_delta': 0.15,
            'wall_collision': 'push',
        },
        'pitch': {
            'length': 2.8, 'width': 2.44,
            'goal_width': 0.75, 'goal_depth': 0.19,
            'goal_area_width': 1.03, 'goal_area_depth': 0.19,
            'penalty_area_width': 1.5, 'penalty_area_depth': 0.47,
            'penalty_spot_distance': 0.38, 'center_circle_radius': 0.28,
            'corner_arc_radius': 0.09, 'wall_thickness': 0.04,
            'wall_mode': 'open',
        },
        'ball': {
            'radius': 0.0335, 'mass': 0.0577, 'gravity': 9.81,
            'rolling_friction': 0.04, 'wall_restitution': 0.745,
            'inertia_factor': 0.6667, 'sliding_friction': 0.23,
            'wall_friction': 0.23, 'robot_friction': 0.23,
            'robot_restitution': 0.5, 'spin_damping': 0.0,
            'placement': 'random',
            'x_range': [-1.3665, 1.3665], 'y_range': [-1.1865, 1.1865],
        },
        'teams': {
            'blue': {
                'num_robots': 1, 'placement': 'random',
                'x_range': [-1.3296, 1.3296], 'y_range': [-1.1496, 1.1496],
                'q_range': [-3.14159, 3.14159],
            },
            'yellow': {'num_robots': 0},
        },
        'sensors': {
            'angles': [-2.53, -1.571, -0.785, -0.175, 0.175, 0.785, 1.571, 2.53],
            'max_range': 0.25, 'min_range': 0.005,
            'noise': {
                'enabled': sensor_noise, 'model': 'correlated',
                'relative_std': 0.04, 'bias_std': 0.015, 'rho': 0.92,
            },
            'normalization': 'minmax',
        },
        'vision': {
            'mode': 'frontal', 'horizontal_fov_deg': 131.0,
            'ball_max_range': 1.9, 'goal_max_range': 3.6,
            'noise': {
                'enabled': vision_noise, 'model': 'correlated_close_range',
                'rho': 0.95,
                'ball_dist_bias_std_m': 0.015, 'ball_angle_bias_std_rad': 0.01,
                'goal_dist_bias_std_m': 0.02, 'goal_angle_bias_std_rad': 0.006,
                'ball_dist_std_far_m': 0.1, 'ball_dist_std_near_m': 0.16,
                'ball_angle_std_far_rad': 0.015, 'ball_angle_std_near_rad': 0.07,
                'goal_dist_std_m': 0.08, 'goal_angle_std_rad': 0.012,
                'near_start': 0.3, 'near_full': 0.12,
                'ball_dropout_far': 0.06, 'ball_dropout_near': 0.22,
                'goal_dropout_far': 0.01, 'goal_dropout_near': 0.03,
            },
        },
        'simulation': {
            'global_seed': 0, 'max_steps': 500,
            'timestep': 0.1, 'physics_timestep': 0.01,
            'num_trials': 1, 'fitness_function': 'penalty_sparse',
            'trajectory_stride': 1,
        },
        'neural_network': {
            'type': 'elman', 'hidden_size': 5,
            'output_size': 2, 'vision_inputs': 4,
            'activation': 'sigmoid', 'wheel_output_mapping': 'floreano',
        },
        'curriculum': {'enabled': False},
        'challenge': {'enabled': False},
        'motor': {'noise': {'enabled': False}},
        'fitness_params': {},
    }


def _get_jax_params(config):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        return load_config_jax(tmp)
    finally:
        os.unlink(tmp)


# ═══════════════════════════════════════════════════
# TIER 2.1 — Sensor noise distribution
# ═══════════════════════════════════════════════════

def test_2_1_sensor_noise(num_trials=1000):
    """Compare IR sensor noise distributions at ball distance 10cm."""
    print("=== Tier 2.1: Sensor noise distribution ===")
    print(f"  {num_trials} trials, robot at origin, ball at (0.10, 0), sensor noise ON")

    config = _make_config(sensor_noise=True, vision_noise=False)
    params = _get_jax_params(config)

    # Forward sensor index: angle +0.175 rad → index 4
    fwd_idx = 4  # 0.175 rad sensor

    # ── NumPy ──
    np_readings = []
    sim = Simulator(config)
    hl = config['pitch']['length'] / 2.0

    for i in range(num_trials):
        sim.reset(random_seed=i)
        sim.robots[0].x, sim.robots[0].y, sim.robots[0].q = 0.0, 0.0, 0.0
        sim.ball.x, sim.ball.y = 0.10, 0.0
        inputs = sim.robots[0].get_all_inputs([], sim.ball, hl)
        np_readings.append(inputs[fwd_idx])

    np_readings = np.array(np_readings)

    # ── JAX ──
    jax_readings = []

    jax_robot = RobotState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
    jax_ball = BallState(
        x=jnp.float32(0.10), y=jnp.float32(0.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

    for i in range(num_trials):
        key = jax.random.PRNGKey(i)
        noise_state = init_noise(key, params)
        raw = get_ir_readings(jax_robot, jax_ball, params)
        k_sensor = jax.random.split(key, 2)[1]
        noisy, _ = apply_sensor_noise(raw, noise_state, k_sensor, params)
        clipped = jnp.clip(noisy, params.sensor_min, params.sensor_max)
        normalized = _normalize_sensors(clipped, params)
        jax_readings.append(float(normalized[fwd_idx]))

    jax_readings = np.array(jax_readings)

    # ── Compare ──
    np_mean, np_std = np_readings.mean(), np_readings.std()
    jax_mean, jax_std = jax_readings.mean(), jax_readings.std()
    se = np_std / np.sqrt(num_trials)  # standard error

    print(f"\n  {'':>12s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'NumPy':>12s} {np_mean:10.6f} {np_std:10.6f} {np_readings.min():10.6f} {np_readings.max():10.6f}")
    print(f"  {'JAX':>12s} {jax_mean:10.6f} {jax_std:10.6f} {jax_readings.min():10.6f} {jax_readings.max():10.6f}")
    print(f"  {'Diff':>12s} {abs(np_mean-jax_mean):10.6f} {abs(np_std-jax_std):10.6f}")
    print(f"  Std error: {se:.6f}")
    print(f"  Mean diff / SE: {abs(np_mean - jax_mean) / max(se, 1e-10):.2f}")

    # KS test
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(np_readings, jax_readings)
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")
    print(f"  KS equivalent (p > 0.01): {'YES' if ks_p > 0.01 else 'NO'}")

    mean_ok = abs(np_mean - jax_mean) < 2 * se  # within 2 standard errors
    std_ok = abs(np_std - jax_std) / max(np_std, 1e-10) < 0.10  # within 10%
    ok = mean_ok and std_ok
    print(f"  Mean within 2 SE: {'YES' if mean_ok else 'NO'}")
    print(f"  Std within 10%:   {'YES' if std_ok else 'NO'}")
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Save histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(np_readings.min(), jax_readings.min()),
                           max(np_readings.max(), jax_readings.max()), 50)
        ax.hist(np_readings, bins=bins, alpha=0.5, label=f'NumPy (μ={np_mean:.4f}, σ={np_std:.4f})', density=True)
        ax.hist(jax_readings, bins=bins, alpha=0.5, label=f'JAX (μ={jax_mean:.4f}, σ={jax_std:.4f})', density=True)
        ax.set_xlabel('Normalized IR reading (forward sensor, ball at 10cm)')
        ax.set_ylabel('Density')
        ax.set_title(f'Tier 2.1: Sensor Noise Distribution (n={num_trials}, KS p={ks_p:.3f})')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = 'jax_sim/tests/tier2_1_sensor_noise.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {save_path}")
    except ImportError:
        pass

    return ok


# ═══════════════════════════════════════════════════
# TIER 2.2 — Vision noise close-range degradation
# ═══════════════════════════════════════════════════

def test_2_2_vision_noise_degradation(num_trials=500):
    """Compare vision noise std at varying ball distances."""
    print("\n=== Tier 2.2: Vision noise close-range degradation ===")
    print(f"  {num_trials} trials per distance")

    config = _make_config(sensor_noise=False, vision_noise=True)
    params = _get_jax_params(config)
    hl = config['pitch']['length'] / 2.0

    distances = [0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"\n  {'Dist':>6s}  {'NP mean':>9s} {'NP std':>9s}  {'JAX mean':>9s} {'JAX std':>9s}  {'Δstd':>8s}")
    print("  " + "-" * 60)

    all_ok = True
    np_stds = []
    jax_stds = []

    for ball_dist in distances:
        # ── NumPy ──
        np_dists_measured = []
        sim = Simulator(config)
        for i in range(num_trials):
            sim.reset(random_seed=i)
            sim.robots[0].x, sim.robots[0].y, sim.robots[0].q = 0.0, 0.0, 0.0
            sim.ball.x, sim.ball.y = ball_dist, 0.0
            inputs = sim.robots[0].get_all_inputs([], sim.ball, hl)
            np_dists_measured.append(inputs[8])  # ball_dist_norm (vision index 0)

        np_arr = np.array(np_dists_measured)

        # ── JAX ──
        jax_dists_measured = []
        from jax_sim.sensors import get_vision_inputs
        from jax_sim.noise import apply_vision_noise

        jax_robot = RobotState(
            x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
            left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
        jax_ball = BallState(
            x=jnp.float32(ball_dist), y=jnp.float32(0.0),
            vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

        for i in range(num_trials):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            noise_state = init_noise(k1, params)

            vision_raw, bv, gv, bd_raw, ba_raw, gd_raw, ga_raw = \
                get_vision_inputs(jax_robot, jax_ball, hl, params)

            bd, ba, gd, ga, bv2, gv2, _ = apply_vision_noise(
                bd_raw, ba_raw, gd_raw, ga_raw,
                bv, gv, noise_state, k2, params)

            bd_n = jnp.clip(bd, 0.0, params.ball_max_range) / params.ball_max_range
            bd_n = jnp.where(bv2, bd_n, 1.0)
            jax_dists_measured.append(float(bd_n))

        jax_arr = np.array(jax_dists_measured)

        np_stds.append(np_arr.std())
        jax_stds.append(jax_arr.std())

        std_diff = abs(np_arr.std() - jax_arr.std())
        std_ok = std_diff / max(np_arr.std(), 1e-10) < 0.20  # within 20%
        if not std_ok:
            all_ok = False

        print(f"  {ball_dist:6.2f}  {np_arr.mean():9.5f} {np_arr.std():9.5f}  "
              f"{jax_arr.mean():9.5f} {jax_arr.std():9.5f}  {std_diff:8.5f} "
              f"{'OK' if std_ok else 'DIFF'}")

    print(f"\n  {'PASS' if all_ok else 'FAIL'}")

    # Save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(distances, np_stds, 'ro-', label='NumPy', markersize=8)
        ax.plot(distances, jax_stds, 'bs-', label='JAX', markersize=8)
        ax.set_xlabel('Ball distance (m)')
        ax.set_ylabel('Std of ball_dist_norm reading')
        ax.set_title(f'Tier 2.2: Vision Noise Degradation (n={num_trials} per distance)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = 'jax_sim/tests/tier2_2_vision_degradation.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {save_path}")
    except ImportError:
        pass

    return all_ok


# ═══════════════════════════════════════════════════
# TIER 2.3 — Vision dropout rate
# ═══════════════════════════════════════════════════

def test_2_3_dropout(num_trials=1000):
    """Compare vision dropout rate at varying distances."""
    print(f"\n=== Tier 2.3: Vision dropout rate ===")
    print(f"  {num_trials} trials per distance")

    config = _make_config(sensor_noise=False, vision_noise=True)
    params = _get_jax_params(config)
    hl = config['pitch']['length'] / 2.0

    distances = [0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"\n  {'Dist':>6s}  {'NP dropout%':>11s}  {'JAX dropout%':>12s}  {'Δ':>6s}")
    print("  " + "-" * 45)

    all_ok = True

    for ball_dist in distances:
        # ── NumPy ──
        np_invisible = 0
        sim = Simulator(config)
        for i in range(num_trials):
            sim.reset(random_seed=i)
            sim.robots[0].x, sim.robots[0].y, sim.robots[0].q = 0.0, 0.0, 0.0
            sim.ball.x, sim.ball.y = ball_dist, 0.0
            inputs = sim.robots[0].get_all_inputs([], sim.ball, hl)
            # If ball_dist_norm == 1.0 (max range sentinel), ball was dropped
            if abs(inputs[8] - 1.0) < 1e-6:
                np_invisible += 1

        np_rate = np_invisible / num_trials

        # ── JAX ──
        jax_invisible = 0
        from jax_sim.sensors import get_vision_inputs
        from jax_sim.noise import apply_vision_noise

        jax_robot = RobotState(
            x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
            left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
        jax_ball = BallState(
            x=jnp.float32(ball_dist), y=jnp.float32(0.0),
            vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

        for i in range(num_trials):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            noise_state = init_noise(k1, params)

            _, bv, _, bd_raw, ba_raw, gd_raw, ga_raw = \
                get_vision_inputs(jax_robot, jax_ball, hl, params)

            _, _, _, _, bv2, _, _ = apply_vision_noise(
                bd_raw, ba_raw, gd_raw, ga_raw,
                bv, jnp.bool_(True), noise_state, k2, params)

            if not bool(bv2):
                jax_invisible += 1

        jax_rate = jax_invisible / num_trials

        diff = abs(np_rate - jax_rate)
        ok = diff < 0.05  # within 5 percentage points
        if not ok:
            all_ok = False

        print(f"  {ball_dist:6.2f}  {np_rate:10.1%}  {jax_rate:11.1%}  {diff:5.1%} "
              f"{'OK' if ok else 'DIFF'}")

    print(f"\n  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TIER 2 — NOISE DISTRIBUTION TESTS")
    print("=" * 60)
    print()

    results = []
    results.append(("2.1 Sensor noise", test_2_1_sensor_noise()))
    results.append(("2.2 Vision degradation", test_2_2_vision_noise_degradation()))
    results.append(("2.3 Dropout rate", test_2_3_dropout()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {sum(1 for _, ok in results if ok)}/{len(results)} passed")


if __name__ == '__main__':
    main()
