"""Top-level training loop — loads config, runs evolution, saves results."""

import os
import sys
import time
import json
import yaml
import numpy as np
import jax
import jax.numpy as jnp

from .config import load_config
from .neural_network import num_weights
from .challenge import sample_states_for_generation
from .evaluator import make_evaluate_population
from .evolution import (initialize_population, create_next_generation,
                        create_next_generation_replacement)
from .fitness import penalty_sparse
from .simulator import rollout


def run(config_path, batch_mode=False):
    """Run a full evolutionary experiment.

    Args:
        config_path: path to YAML config file
        batch_mode: if True, suppress plots
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"JAX devices: {jax.devices()}")
    print(f"Config: {config_path}")

    sim_cfg = config.get('simulation', {})
    evo_cfg = config.get('evolution', {})
    challenge_cfg = config.get('challenge', {})

    # Handle random seed
    if sim_cfg.get('random_seed', False):
        global_seed = int(time.time()) % (2 ** 31)
        config['simulation']['global_seed'] = global_seed
        print(f"Random seed mode: global_seed = {global_seed}")
    else:
        global_seed = sim_cfg.get('global_seed', 0)

    np_rng = np.random.RandomState(global_seed)

    # Load static params for JAX
    params = load_config(config_path)

    # Population config
    pop_size = evo_cfg.get('population_size', 128)
    num_gen = evo_cfg.get('num_generations', 500)
    num_trials = sim_cfg.get('num_trials', 5)
    n_weights = num_weights(params.input_size, params.hidden_size, params.output_size)

    # Evolution params
    bounds = evo_cfg.get('weight_bounds', [-8.0, 8.0])
    iwr = evo_cfg.get('initial_weight_range', bounds)
    crossover_rate = evo_cfg.get('crossover_rate', 0.2)
    mutation_rate = evo_cfg.get('mutation_rate', 0.15)
    mutation_scale = evo_cfg.get('mutation_scale', 0.2)
    mutation_type = evo_cfg.get('mutation_type', 'gaussian')
    tournament_size = evo_cfg.get('tournament_size', 2)
    num_elite = evo_cfg.get('elitism', 4)
    total_gen = challenge_cfg.get('total_generations', num_gen)

    print(f"Population: {pop_size}, Generations: {num_gen}, Weights: {n_weights}")
    print(f"Bounds: {bounds}, IWR: {iwr}")
    print(f"MutR: {mutation_rate}, MutS: {mutation_scale}, MutT: {mutation_type}, CrR: {crossover_rate}")

    # Results directory
    results_dir = config.get('results_dir', 'results/jax_run')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize population
    init_key = jax.random.PRNGKey(global_seed)
    population = initialize_population(
        init_key, pop_size, n_weights, iwr[0], iwr[1])

    # JIT-compile evaluator
    print("Compiling evaluation function (first call will be slow)...")
    evaluate_fn = make_evaluate_population(params)

    # Tracking
    generation_stats = []
    best_fitness_history = []
    true_goal_rates = []
    checkpoint_interval = 10
    checkpoints = []  # list of (label, weights) — saved every checkpoint_interval gens

    t_start = time.time()

    for gen in range(num_gen):
        gen_start = time.time()

        # Sample initial states for this generation
        init_states, p = sample_states_for_generation(
            pop_size, num_trials, gen, total_gen, config, np_rng)
        init_states_jax = jnp.array(init_states)

        # Generate keys for this generation
        gen_key = jax.random.PRNGKey(
            (global_seed * 10_000_000 + gen * 100_000) % (2 ** 31))
        trial_keys = jax.random.split(gen_key, pop_size * num_trials)
        trial_keys = trial_keys.reshape(pop_size, num_trials, 2)

        # Evaluate
        fitness, goal_rates = evaluate_fn(population, init_states_jax, trial_keys)
        fitness = np.array(fitness)
        goal_rates_np = np.array(goal_rates)

        # Stats
        best_idx = np.argmax(fitness)
        gen_best = fitness[best_idx]
        gen_mean = np.mean(fitness)
        gen_std = np.std(fitness)
        gen_worst = np.min(fitness)

        generation_stats.append({
            'best': float(gen_best),
            'mean': float(gen_mean),
            'std': float(gen_std),
            'worst': float(gen_worst),
        })
        best_fitness_history.append(float(gen_best))

        # Save checkpoint every N generations and at the last generation
        is_checkpoint = (gen % checkpoint_interval == 0) or (gen == num_gen - 1)

        if is_checkpoint:
            tr_pct = int(np.mean(goal_rates_np) * 100)
            label = f"gen{gen+1}_tr{tr_pct}%"
            checkpoints.append((label, np.array(population[best_idx])))

            true_goal_rates.append({
                'gen': gen + 1,
                'goal_rate': float(np.mean(goal_rates_np)),
            })

        gen_time = time.time() - gen_start

        if is_checkpoint:
            print(f"Gen {gen+1:>4d}/{num_gen}  "
                  f"best={gen_best:.2f}  mean={gen_mean:.2f}  "
                  f"p={p:.3f}  goal_rate={np.mean(goal_rates_np):.1%}  "
                  f"time={gen_time:.1f}s")

        # Create next generation
        if gen < num_gen - 1:
            ga_key = jax.random.PRNGKey(global_seed + gen + 1)
            if mutation_type == 'replacement':
                population = create_next_generation_replacement(
                    ga_key, population, jnp.array(fitness),
                    pop_size, num_elite, tournament_size,
                    crossover_rate, mutation_rate,
                    (bounds[0], bounds[1]))
            else:
                population = create_next_generation(
                    ga_key, population, jnp.array(fitness),
                    pop_size, num_elite, tournament_size,
                    crossover_rate, mutation_rate, mutation_scale,
                    (bounds[0], bounds[1]))

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Add top N from last generation as candidates
    top_n = 10
    top_idx = np.argsort(fitness)[-top_n:][::-1]
    last_gen_candidates = []
    for rank, idx in enumerate(top_idx):
        label = f"last_gen_rank{rank}"
        last_gen_candidates.append((label, np.array(population[idx])))

    # Save generation stats and true goal rates
    np.save(os.path.join(results_dir, 'fitness_history.npy'),
            np.array(best_fitness_history))

    with open(os.path.join(results_dir, 'generation_stats.json'), 'w') as f:
        json.dump(generation_stats, f, indent=2)

    with open(os.path.join(results_dir, 'true_goal_rates.json'), 'w') as f:
        json.dump(true_goal_rates, f, indent=2)

    # Save top N individuals
    top_weights = np.array(population[top_idx])
    np.save(os.path.join(results_dir, 'top_weights.npy'), top_weights)

    # Save config (with resolved seed)
    config_dest = os.path.join(results_dir, os.path.basename(config_path))
    with open(config_dest, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Results saved to {results_dir}/")

    # Validate ALL candidates (last gen top + checkpoints) on 50 unseen scenarios
    all_candidates = last_gen_candidates + checkpoints
    print(f"\n  Validating {len(all_candidates)} candidates: "
          f"{len(last_gen_candidates)} from last gen + {len(checkpoints)} checkpoints")

    if params.validate_in_jax:
        print("  Validation mode: JAX")
        _validate_candidates_jax(all_candidates, params, config, results_dir)
    else:
        print("  Validation mode: NumPy")
        _validate_candidates_numpy(all_candidates, params, config, results_dir)


def _validate_candidates_numpy(candidates, params, config, results_dir):
    """Validate using NumPy Simulator (original behavior).

    Uses the NumPy Simulator with challenge DISABLED and uniform random
    placement, matching the NumPy main.py validation exactly.
    Seeds: 100000..100049 (deterministic, same across all runs).
    """
    from core.simulator import Simulator as NumpySimulator

    num_val = 50
    seed_offset = 100000

    # Create NumPy simulator with challenge/curriculum disabled
    val_config = dict(config)
    if 'challenge' in val_config:
        val_config['challenge'] = dict(val_config['challenge'])
        val_config['challenge']['enabled'] = False
    if 'curriculum' in val_config:
        val_config['curriculum'] = dict(val_config['curriculum'])
        val_config['curriculum']['enabled'] = False

    simulator = NumpySimulator(val_config)
    simulator._state_pool = None

    print(f"\n  Validating {len(candidates)} candidates on {num_val} unseen scenarios...")

    candidate_labels = []
    candidate_goals = []
    best_goals = -1
    best_label = ""
    best_weights = None

    for label, weights in candidates:
        goals = 0
        for i in range(num_val):
            seed = seed_offset + i
            metrics = simulator.run(weights, random_seed=seed)
            if metrics.get('goal_scored') == metrics.get('target_goal_side'):
                goals += 1

        pct = int(goals / num_val * 100)
        print(f"    {label}: {goals}/{num_val} goals ({pct}%)")

        candidate_labels.append(label)
        candidate_goals.append(goals)

        if goals > best_goals:
            best_goals = goals
            best_label = label
            best_weights = weights

    # Find overall best candidate
    best_idx = np.argmax(candidate_goals)
    selected_label = candidate_labels[best_idx]
    selected_goals = candidate_goals[best_idx]
    print(f"  → Selected (overall best): {selected_label} ({selected_goals}/{num_val} goals)")

    # Find best from last generation (labels starting with "last_gen_")
    last_gen_mask = [l.startswith("last_gen_") for l in candidate_labels]
    last_gen_goals = [g if m else -1 for g, m in zip(candidate_goals, last_gen_mask)]
    best_last_gen_idx = np.argmax(last_gen_goals)
    best_last_gen_label = candidate_labels[best_last_gen_idx]
    best_last_gen_goals = candidate_goals[best_last_gen_idx]
    print(f"  → Best last gen: {best_last_gen_label} ({best_last_gen_goals}/{num_val} goals)")

    # Save overall best weights
    np.save(os.path.join(results_dir, 'best_weights.npy'), best_weights)

    # Save best last-gen weights separately
    for label, weights in candidates:
        if label == best_last_gen_label:
            np.save(os.path.join(results_dir, 'best_last_gen_weights.npy'), weights)
            break

    # Save trial details (same format as NumPy version)
    trial_details = {
        'validation': {'num_scenarios': num_val, 'seed_range': [100000, 100000 + num_val - 1]},
        'selected_candidate': int(best_idx),
        'selected_label': selected_label,
        'candidate_goals': [int(g) for g in candidate_goals],
        'candidate_labels': candidate_labels,
    }
    with open(os.path.join(results_dir, 'trial_details.json'), 'w') as f:
        json.dump(trial_details, f, indent=2)

    # Save validation summary
    goal_rate = selected_goals / num_val
    last_gen_rate = best_last_gen_goals / num_val
    summary = (
        f"{'=' * 60}\n"
        f"  Validation: testing on {num_val} unseen scenarios\n"
        f"  Validation seeds: [100000..{100000 + num_val - 1}]\n"
        f"  Selected: {selected_label} (candidate {best_idx} of {len(candidates)})\n"
        f"{'=' * 60}\n\n"
        f"  Overall best:       {selected_goals}/{num_val} ({int(goal_rate * 100)}%)  [{selected_label}]\n"
        f"  Best last gen:      {best_last_gen_goals}/{num_val} ({int(last_gen_rate * 100)}%)  [{best_last_gen_label}]\n"
        f"  Goals scored:       {selected_goals}/{num_val} ({int(goal_rate * 100)}%)\n"
        f"{'=' * 60}\n"
    )
    with open(os.path.join(results_dir, 'validation_summary.txt'), 'w') as f:
        f.write(summary)


def _validate_candidates_jax(candidates, params, config, results_dir):
    """Validate entirely in JAX — no NumPy simulator dependency.

    Samples uniform initial states with NumPy RNG (deterministic),
    then runs all rollouts through the JAX simulation engine.
    """
    from .challenge import sample_uniform_states

    num_val = 50
    seed_offset = 100000

    # Sample uniform states (same RNG seed for reproducibility)
    val_rng = np.random.RandomState(42)
    states = sample_uniform_states(num_val, config, val_rng)

    print(f"\n  Validating {len(candidates)} candidates on {num_val} unseen scenarios (JAX)...")

    candidate_labels = []
    candidate_goals = []
    best_goals = -1
    best_label = ""
    best_weights = None

    # JIT-compile a single rollout
    @jax.jit
    def eval_one(weights_jax, rx, ry, rq, bx, by, key):
        final = rollout(weights_jax, rx, ry, rq, bx, by, key, params)
        return final.goal_scored

    dtype = jnp.float64 if params.use_x64 else jnp.float32

    for label, weights in candidates:
        weights_jax = jnp.array(weights, dtype=dtype)
        goals = 0
        for i in range(num_val):
            rx, ry, rq, bx, by = states[i]
            key = jax.random.PRNGKey(seed_offset + i)
            scored = int(eval_one(weights_jax, rx, ry, rq, bx, by, key))
            if scored == 1:  # right goal = target
                goals += 1

        pct = int(goals / num_val * 100)
        print(f"    {label}: {goals}/{num_val} goals ({pct}%)")

        candidate_labels.append(label)
        candidate_goals.append(goals)

        if goals > best_goals:
            best_goals = goals
            best_label = label
            best_weights = weights

    # Find overall best
    best_idx = np.argmax(candidate_goals)
    selected_label = candidate_labels[best_idx]
    selected_goals = candidate_goals[best_idx]
    print(f"  → Selected (overall best): {selected_label} ({selected_goals}/{num_val} goals)")

    # Find best from last generation
    last_gen_mask = [l.startswith("last_gen_") for l in candidate_labels]
    last_gen_goals = [g if m else -1 for g, m in zip(candidate_goals, last_gen_mask)]
    best_last_gen_idx = np.argmax(last_gen_goals)
    best_last_gen_label = candidate_labels[best_last_gen_idx]
    best_last_gen_goals = candidate_goals[best_last_gen_idx]
    print(f"  → Best last gen: {best_last_gen_label} ({best_last_gen_goals}/{num_val} goals)")

    # Save weights
    np.save(os.path.join(results_dir, 'best_weights.npy'), best_weights)
    for label, weights in candidates:
        if label == best_last_gen_label:
            np.save(os.path.join(results_dir, 'best_last_gen_weights.npy'), weights)
            break

    # Save trial details
    trial_details = {
        'validation': {
            'num_scenarios': num_val,
            'seed_range': [seed_offset, seed_offset + num_val - 1],
            'mode': 'jax',
        },
        'selected_candidate': int(best_idx),
        'selected_label': selected_label,
        'candidate_goals': [int(g) for g in candidate_goals],
        'candidate_labels': candidate_labels,
    }
    with open(os.path.join(results_dir, 'trial_details.json'), 'w') as f:
        json.dump(trial_details, f, indent=2)

    # Save validation summary
    goal_rate = selected_goals / num_val
    val_std = np.sqrt(goal_rate * (1 - goal_rate))
    last_gen_rate = best_last_gen_goals / num_val
    summary = (
        f"{'=' * 60}\n"
        f"  Validation: testing on {num_val} unseen scenarios (JAX)\n"
        f"  Validation seeds: [{seed_offset}..{seed_offset + num_val - 1}]\n"
        f"  Selected: {selected_label} (candidate {best_idx} of {len(candidates)})\n"
        f"{'=' * 60}\n\n"
        f"  Overall best:       {selected_goals}/{num_val} ({int(goal_rate * 100)}%)  [{selected_label}]\n"
        f"  Best last gen:      {best_last_gen_goals}/{num_val} ({int(last_gen_rate * 100)}%)  [{best_last_gen_label}]\n"
        f"  Goals scored:       {selected_goals}/{num_val} ({int(goal_rate * 100)}%)\n"
        f"{'=' * 60}\n"
    )
    with open(os.path.join(results_dir, 'validation_summary.txt'), 'w') as f:
        f.write(summary)


def main():
    config_path = 'config/config.yaml'
    batch_mode = '--batch' in sys.argv
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--config' and i < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif arg.startswith('--config='):
            config_path = arg.split('=', 1)[1]

    run(config_path, batch_mode)


if __name__ == '__main__':
    main()
