"""Tier 4.1 — Fitness trajectory shape comparison.

Runs 5 short evolutions (50 generations, pop=64) in both NumPy and JAX.
Compares best-fitness curves across seeds.

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_tier4_1_fitness_curves.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import json
import numpy as np
import yaml
import tempfile

# ─── Config ───

NUM_SEEDS = 5
NUM_GENERATIONS = 50
POP_SIZE = 64
NUM_TRIALS = 5
MAX_STEPS = 500

CONFIG = {
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
        'x_range': [-1.3665, 1.3665],
        'y_range': [-1.1865, 1.1865],
    },
    'teams': {
        'blue': {
            'num_robots': 1, 'placement': 'random',
            'x_range': [-1.3296, 1.3296],
            'y_range': [-1.1496, 1.1496],
            'q_range': [-3.14159, 3.14159],
        },
        'yellow': {'num_robots': 0},
    },
    'sensors': {
        'angles': [-2.53, -1.571, -0.785, -0.175, 0.175, 0.785, 1.571, 2.53],
        'max_range': 0.25, 'min_range': 0.005,
        'noise': {'enabled': True, 'model': 'correlated',
                  'relative_std': 0.04, 'bias_std': 0.015, 'rho': 0.92},
        'normalization': 'minmax',
    },
    'vision': {
        'mode': 'frontal', 'horizontal_fov_deg': 131.0,
        'ball_max_range': 1.9, 'goal_max_range': 3.6,
        'noise': {
            'enabled': True, 'model': 'correlated_close_range', 'rho': 0.95,
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
        'global_seed': 0, 'max_steps': MAX_STEPS,
        'timestep': 0.1, 'physics_timestep': 0.01,
        'num_trials': NUM_TRIALS, 'fitness_function': 'penalty_sparse',
        'trajectory_stride': 1,
    },
    'neural_network': {
        'type': 'elman', 'hidden_size': 5,
        'output_size': 2, 'vision_inputs': 4,
        'activation': 'sigmoid', 'wheel_output_mapping': 'floreano',
    },
    'curriculum': {'enabled': False},
    'challenge': {
        'enabled': True, 'parameter': 'forward_l4',
        'schedule': 'linear', 'total_generations': NUM_GENERATIONS,
        'p_min': 0.022, 'mode': 'cumulative', 'k': 1,
        'area_correction': False,
    },
    'motor': {'noise': {'enabled': False}},
    'fitness_params': {'goal_reward': 1.0, 'time_bonus': False},
    'evolution': {
        'algorithm': 'ga', 'population_size': POP_SIZE,
        'num_generations': NUM_GENERATIONS,
        'crossover_rate': 0.2, 'crossover_type': 'single_point',
        'weight_bounds': [-4.0, 4.0],
        'initial_weight_range': [-0.5, 0.5],
        'elitism': 2, 'tournament_size': 2,
        'mutation_rate': 0.15, 'mutation_scale': 0.2,
        'mutation_type': 'gaussian', 'immigrant_frac': 0.0,
        'l2_lambda': 0.0, 'num_workers': 8,
    },
    'seeding': {'enabled': False},
}


# ─── JAX evolution ───

def run_jax_evolution(seed):
    """Run one JAX evolution, return best-fitness-per-gen array."""
    import jax
    import jax.numpy as jnp
    from jax_sim.config import load_config as load_config_jax
    from jax_sim.neural_network import num_weights
    from jax_sim.evaluator import make_evaluate_population
    from jax_sim.challenge import sample_states_for_generation
    from jax_sim.evolution import initialize_population, create_next_generation

    config = dict(CONFIG)
    config['simulation'] = dict(config['simulation'])
    config['simulation']['global_seed'] = seed

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        params = load_config_jax(tmp)
    finally:
        os.unlink(tmp)

    np_rng = np.random.RandomState(seed)
    n = num_weights(params.input_size, params.hidden_size, params.output_size)

    iwr = config['evolution']['initial_weight_range']
    pop = initialize_population(jax.random.PRNGKey(seed), POP_SIZE, n, iwr[0], iwr[1])

    evaluate_fn = make_evaluate_population(params)

    bounds = config['evolution']['weight_bounds']
    best_per_gen = []

    for gen in range(NUM_GENERATIONS):
        states, p = sample_states_for_generation(
            POP_SIZE, NUM_TRIALS, gen, NUM_GENERATIONS, config, np_rng)
        gen_key = jax.random.PRNGKey((seed * 10_000_000 + gen * 100_000) % (2**31))
        trial_keys = jax.random.split(gen_key, POP_SIZE * NUM_TRIALS).reshape(POP_SIZE, NUM_TRIALS, 2)

        fitness, _ = evaluate_fn(pop, jnp.array(states), trial_keys)
        fitness_np = np.array(fitness)
        best_per_gen.append(float(np.max(fitness_np)))

        if gen < NUM_GENERATIONS - 1:
            ga_key = jax.random.PRNGKey(seed + gen + 1)
            pop = create_next_generation(
                ga_key, pop, jnp.array(fitness_np),
                POP_SIZE, 2, 2, 0.2, 0.15, 0.2, (bounds[0], bounds[1]))

    return np.array(best_per_gen)


# ─── NumPy evolution ───

_np_worker_sim = None
_np_worker_fitness_fn = None
_np_worker_seed = None
_np_worker_gen = None


def _np_init_worker(config, global_seed):
    global _np_worker_sim, _np_worker_fitness_fn, _np_worker_seed
    from core.simulator import Simulator
    from evolution.fitness_functions import get_fitness_function
    _np_worker_sim = Simulator(config)
    _np_worker_fitness_fn = get_fitness_function(
        config['simulation']['fitness_function'], config)
    _np_worker_seed = global_seed


def _np_eval_single(args):
    ind_idx, weights, gen = args
    total = 0.0
    _np_worker_sim.set_generation(gen)
    for trial in range(NUM_TRIALS):
        s = (_np_worker_seed * 10_000_000 + gen * 100_000 + ind_idx * 1000 + trial) % (2**31)
        metrics = _np_worker_sim.run(weights, random_seed=s)
        total += _np_worker_fitness_fn(metrics)
    return total / NUM_TRIALS


def run_numpy_evolution(seed):
    """Run one NumPy evolution with multiprocessing, return best-fitness-per-gen array."""
    from evolution.evolutionary_algorithm import EvolutionaryAlgorithm, EvolutionConfig
    from multiprocessing import Pool

    config = dict(CONFIG)
    config['simulation'] = dict(config['simulation'])
    config['simulation']['global_seed'] = seed
    np.random.seed(seed)

    evo_cfg = config['evolution']
    bounds = evo_cfg['weight_bounds']
    iwr = evo_cfg.get('initial_weight_range', bounds)
    num_workers = evo_cfg.get('num_workers', 4)

    ec = EvolutionConfig(
        population_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        mutation_rate=evo_cfg['mutation_rate'],
        mutation_scale=evo_cfg['mutation_scale'],
        crossover_rate=evo_cfg['crossover_rate'],
        bounds=tuple(bounds),
        elitism=evo_cfg['elitism'],
        tournament_size=evo_cfg['tournament_size'],
        crossover_type=evo_cfg['crossover_type'],
        mutation_type=evo_cfg['mutation_type'],
        initial_weight_range=tuple(iwr) if iwr else None,
    )

    from jax_sim.neural_network import num_weights
    n = num_weights(12, 5, 2)
    ea = EvolutionaryAlgorithm(ec)
    ea.initialize_population(n)

    pool = Pool(processes=num_workers,
                initializer=_np_init_worker,
                initargs=(config, seed))

    best_per_gen = []

    for gen in range(NUM_GENERATIONS):
        args = [(i, ea.population[i], gen) for i in range(POP_SIZE)]
        fitness_vals = pool.map(_np_eval_single, args)

        ea.fitness_values = np.array(fitness_vals)
        best_per_gen.append(float(np.max(ea.fitness_values)))

        if gen < NUM_GENERATIONS - 1:
            ea.create_next_generation()

    pool.close()
    pool.join()

    return np.array(best_per_gen)


# ─── Main ───

def main():
    print("=" * 60)
    print("  Tier 4.1 — Fitness trajectory comparison")
    print(f"  {NUM_SEEDS} seeds × {NUM_GENERATIONS} generations × pop {POP_SIZE}")
    print("=" * 60)
    print()

    jax_curves = []
    numpy_curves = []

    for seed in range(NUM_SEEDS):
        print(f"--- Seed {seed} ---")

        # JAX
        t0 = time.time()
        jax_curve = run_jax_evolution(seed)
        jax_time = time.time() - t0
        jax_curves.append(jax_curve)
        print(f"  JAX:   {jax_time:.1f}s, final best={jax_curve[-1]:.2f}")

        # NumPy
        t0 = time.time()
        numpy_curve = run_numpy_evolution(seed)
        numpy_time = time.time() - t0
        numpy_curves.append(numpy_curve)
        print(f"  NumPy: {numpy_time:.1f}s, final best={numpy_curve[-1]:.2f}")
        print(f"  Speedup: {numpy_time / max(jax_time, 0.1):.1f}x")
        print()

    # Compare
    jax_arr = np.array(jax_curves)     # (NUM_SEEDS, NUM_GENERATIONS)
    np_arr = np.array(numpy_curves)

    jax_mean = jax_arr.mean(axis=0)
    np_mean = np_arr.mean(axis=0)

    jax_finals = jax_arr[:, -1]
    np_finals = np_arr[:, -1]

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  JAX final best:   mean={jax_finals.mean():.3f} ± {jax_finals.std():.3f}  "
          f"range=[{jax_finals.min():.2f}, {jax_finals.max():.2f}]")
    print(f"  NumPy final best: mean={np_finals.mean():.3f} ± {np_finals.std():.3f}  "
          f"range=[{np_finals.min():.2f}, {np_finals.max():.2f}]")
    print(f"  Difference in means: {abs(jax_finals.mean() - np_finals.mean()):.3f}")

    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(jax_finals, np_finals, alternative='two-sided')
    print(f"  Mann-Whitney U: stat={stat:.1f}, p={pval:.4f}")
    print(f"  Equivalent (p > 0.05): {'YES' if pval > 0.05 else 'NO'}")

    # Save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Mean curves
        gens = np.arange(1, NUM_GENERATIONS + 1)
        ax1.plot(gens, jax_mean, 'b-', label='JAX mean', linewidth=2)
        ax1.fill_between(gens, jax_arr.min(0), jax_arr.max(0), alpha=0.15, color='blue')
        ax1.plot(gens, np_mean, 'r-', label='NumPy mean', linewidth=2)
        ax1.fill_between(gens, np_arr.min(0), np_arr.max(0), alpha=0.15, color='red')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Over Generations')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Per-seed final fitness
        seeds = np.arange(NUM_SEEDS)
        width = 0.35
        ax2.bar(seeds - width/2, jax_finals, width, label='JAX', color='blue', alpha=0.7)
        ax2.bar(seeds + width/2, np_finals, width, label='NumPy', color='red', alpha=0.7)
        ax2.set_xlabel('Seed')
        ax2.set_ylabel('Final Best Fitness')
        ax2.set_title(f'Final Best Fitness (p={pval:.3f})')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_path = 'jax_sim/tests/tier4_1_fitness_comparison.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {save_path}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    # Save raw data
    data = {
        'jax_curves': jax_arr.tolist(),
        'numpy_curves': np_arr.tolist(),
        'jax_finals': jax_finals.tolist(),
        'numpy_finals': np_finals.tolist(),
        'mann_whitney_p': float(pval),
    }
    data_path = 'jax_sim/tests/tier4_1_results.json'
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Data saved: {data_path}")


if __name__ == '__main__':
    main()
