"""GA Sanity Checks — Verify evolution operators independently of simulator.

4 tests:
  1. Sphere convergence — does the GA optimize at all?
  2. Elitism monotonic — is best fitness non-decreasing?
  3. Diversity decreases — does selection pressure work?
  4. Mutation distribution — correct rate, scale, mean?

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_ga_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from jax_sim.evolution import initialize_population, create_next_generation, gaussian_mutation


def sphere_fitness(population):
    """Negative sum of squared weights. Optimum at origin = 0."""
    return -jnp.sum(population ** 2, axis=1)


# ═══════════════════════════════════════════════════
# Test 1 — Sphere convergence
# ═══════════════════════════════════════════════════

def test_sphere_convergence():
    pop_size = 64
    num_weights = 128
    bounds = (-8.0, 8.0)
    iwr = (-4.0, 4.0)
    crossover_rate = 0.2
    mutation_rate = 0.15
    mutation_scale = 0.45
    tournament_size = 2
    num_elite = 4
    num_gen = 200

    key = jax.random.PRNGKey(0)
    pop = initialize_population(key, pop_size, num_weights, iwr[0], iwr[1])

    history = []
    for gen in range(num_gen):
        fitness = sphere_fitness(pop)
        best = float(jnp.max(fitness))
        history.append(best)

        if gen % 50 == 0 or gen == num_gen - 1:
            print(f"    Gen {gen:>3d}: best={best:.3f}")

        if gen < num_gen - 1:
            ga_key = jax.random.PRNGKey(gen + 1)
            pop = create_next_generation(
                ga_key, pop, fitness,
                pop_size, num_elite, tournament_size,
                crossover_rate, mutation_rate, mutation_scale,
                bounds)

    initial_best = history[0]
    final_best = history[-1]
    improvement_ratio = abs(initial_best) / max(abs(final_best), 1e-6)

    print(f"  Initial best: {initial_best:.3f}")
    print(f"  Final best:   {final_best:.3f}")
    print(f"  Improvement:  {improvement_ratio:.1f}x")

    pass_no_decline = final_best > initial_best
    pass_threshold = final_best > -10.0
    pass_ratio = improvement_ratio > 10.0

    if pass_no_decline and pass_threshold and pass_ratio:
        print("  PASS")
        return True
    else:
        print(f"  FAIL — decline:{not pass_no_decline} "
              f"threshold:{not pass_threshold} ratio:{not pass_ratio}")
        return False


# ═══════════════════════════════════════════════════
# Test 2 — Elitism monotonic
# ═══════════════════════════════════════════════════

def test_elitism_monotonic():
    pop_size = 64
    num_weights = 128
    bounds = (-8.0, 8.0)
    iwr = (-4.0, 4.0)
    num_gen = 100

    key = jax.random.PRNGKey(42)
    pop = initialize_population(key, pop_size, num_weights, iwr[0], iwr[1])

    declines = []
    prev_best = -np.inf
    for gen in range(num_gen):
        fitness = sphere_fitness(pop)
        current_best = float(jnp.max(fitness))

        if current_best < prev_best - 1e-6:  # tolerance for float noise
            declines.append((gen, prev_best, current_best))

        prev_best = max(prev_best, current_best)  # track running max

        if gen < num_gen - 1:
            ga_key = jax.random.PRNGKey(gen + 1)
            pop = create_next_generation(
                ga_key, pop, fitness,
                pop_size, 4, 2,
                0.2, 0.15, 0.45,
                bounds)

    if not declines:
        print(f"  PASS — best fitness monotonic across {num_gen} generations")
        print(f"  Final best: {prev_best:.4f}")
        return True
    else:
        print(f"  FAIL — best fitness declined {len(declines)} times")
        for gen, prev, curr in declines[:5]:
            print(f"    Gen {gen}: {prev:.6f} -> {curr:.6f} (drop: {prev - curr:.6f})")
        return False


# ═══════════════════════════════════════════════════
# Test 3 — Diversity decreases
# ═══════════════════════════════════════════════════

def test_diversity_decreases():
    pop_size = 64
    num_weights = 128
    bounds = (-8.0, 8.0)
    iwr = (-4.0, 4.0)
    num_gen = 200
    sample_every = 25

    key = jax.random.PRNGKey(123)
    pop = initialize_population(key, pop_size, num_weights, iwr[0], iwr[1])

    diversity_history = []
    for gen in range(num_gen):
        fitness = sphere_fitness(pop)

        if gen % sample_every == 0 or gen == num_gen - 1:
            diversity = float(jnp.std(pop, axis=0).mean())
            diversity_history.append((gen, diversity))

        if gen < num_gen - 1:
            ga_key = jax.random.PRNGKey(gen + 1)
            pop = create_next_generation(
                ga_key, pop, fitness,
                pop_size, 4, 2,
                0.2, 0.15, 0.45,
                bounds)

    initial_div = diversity_history[0][1]
    final_div = diversity_history[-1][1]
    ratio = final_div / max(initial_div, 1e-6)

    print(f"  Diversity trajectory:")
    for gen, div in diversity_history:
        print(f"    Gen {gen:>3d}: {div:.4f}")
    print(f"  Final/initial ratio: {ratio:.3f}")

    pass_decrease = ratio < 0.5
    pass_not_collapsed = final_div > 0.01

    if pass_decrease and pass_not_collapsed:
        print("  PASS — diversity decreased meaningfully without collapse")
        return True
    else:
        print(f"  FAIL — decrease:{not pass_decrease} "
              f"collapsed:{not pass_not_collapsed}")
        return False


# ═══════════════════════════════════════════════════
# Test 4 — Mutation distribution
# ═══════════════════════════════════════════════════

def test_mutation_distribution():
    num_weights = 128
    mutation_rate = 0.15
    mutation_scale = 0.45
    bounds_low = -8.0
    bounds_high = 8.0
    num_trials = 10000

    individual = jnp.zeros(num_weights)

    all_deltas = []
    mutated_mask_counts = []

    for trial in range(num_trials):
        key = jax.random.PRNGKey(trial)
        mutated = gaussian_mutation(
            key, individual, mutation_rate, mutation_scale,
            bounds_low, bounds_high)
        deltas = np.array(mutated - individual)
        all_deltas.append(deltas)
        mutated_mask_counts.append(np.sum(deltas != 0.0))

    all_deltas = np.array(all_deltas)
    mutation_fraction = np.mean(mutated_mask_counts) / num_weights

    nonzero_deltas = all_deltas[all_deltas != 0.0]
    delta_std = np.std(nonzero_deltas)
    delta_mean = np.mean(nonzero_deltas)

    print(f"  Mutation rate:     observed {mutation_fraction:.4f}, "
          f"expected {mutation_rate}")
    print(f"  Mutation scale:    observed {delta_std:.4f}, "
          f"expected {mutation_scale}")
    print(f"  Mutation mean:     observed {delta_mean:+.4f}, "
          f"expected 0.0")

    pass_rate = abs(mutation_fraction - mutation_rate) < 0.02
    pass_scale = abs(delta_std - mutation_scale) / mutation_scale < 0.05
    pass_mean = abs(delta_mean) < 0.02

    if pass_rate and pass_scale and pass_mean:
        print("  PASS")
        return True
    else:
        print(f"  FAIL — rate:{not pass_rate} scale:{not pass_scale} "
              f"mean:{not pass_mean}")
        return False


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  GA SANITY CHECKS")
    print("=" * 60)

    print("\nTest 1 — Sphere convergence")
    r1 = test_sphere_convergence()

    print("\nTest 2 — Elitism monotonic")
    r2 = test_elitism_monotonic()

    print("\nTest 3 — Diversity decreases")
    r3 = test_diversity_decreases()

    print("\nTest 4 — Mutation distribution")
    r4 = test_mutation_distribution()

    print("\n" + "=" * 60)
    print(f"  Summary: {sum([r1, r2, r3, r4])}/4 passed")
    print("=" * 60)
