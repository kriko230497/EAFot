"""JAX-native GA operators — selection, crossover, mutation.

All operations are JIT-compatible and operate on population arrays.
"""

import jax
import jax.numpy as jnp
from functools import partial


def _reflect_bounds(x, low, high):
    """Reflect values into [low, high] using triangle-wave folding.

    Direct translation of the NumPy reference implementation.
    """
    width = high - low
    y = jnp.mod(x - low, 2.0 * width)
    y = jnp.where(y > width, 2.0 * width - y, y)
    return low + y


def tournament_selection(key, population, fitness, tournament_size):
    """Tournament selection for all offspring positions.

    Returns indices of selected parents (pop_size,).
    """
    pop_size = population.shape[0]

    def select_one(key_i):
        candidates = jax.random.randint(key_i, (tournament_size,), 0, pop_size)
        candidate_fitness = fitness[candidates]
        winner = candidates[jnp.argmax(candidate_fitness)]
        return winner

    keys = jax.random.split(key, pop_size)
    return jax.vmap(select_one)(keys)


def single_point_crossover_pair(key, parent1, parent2, crossover_rate):
    """Single-point crossover producing TWO complementary children.

    With probability crossover_rate, perform crossover producing
    (child1, child2). Else return (parent1, parent2).
    """
    k1, k2 = jax.random.split(key)
    do_cross = jax.random.uniform(k1) < crossover_rate
    point = jax.random.randint(k2, (), 1, parent1.shape[0])
    mask = jnp.arange(parent1.shape[0]) < point
    child1 = jnp.where(mask, parent1, parent2)
    child2 = jnp.where(mask, parent2, parent1)
    return (jnp.where(do_cross, child1, parent1),
            jnp.where(do_cross, child2, parent2))


def gaussian_mutation(key, individual, mutation_rate, mutation_scale, low, high):
    """Gaussian mutation with reflection bounds."""
    k1, k2 = jax.random.split(key)
    mask = jax.random.uniform(k1, individual.shape) < mutation_rate
    noise = jax.random.normal(k2, individual.shape) * mutation_scale
    mutated = individual + jnp.where(mask, noise, 0.0)
    return _reflect_bounds(mutated, low, high)


def replacement_mutation(key, individual, mutation_rate, low, high):
    """Replacement mutation — replace genes with uniform random from bounds."""
    k1, k2 = jax.random.split(key)
    mask = jax.random.uniform(k1, individual.shape) < mutation_rate
    replacements = jax.random.uniform(k2, individual.shape, minval=low, maxval=high)
    return jnp.where(mask, replacements, individual)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def create_next_generation(key, population, fitness,
                           pop_size, num_elite, tournament_size,
                           crossover_rate, mutation_rate, mutation_scale,
                           bounds_tuple):
    """Create next generation using GA operators.

    Args:
        key: PRNGKey
        population: (pop_size, num_weights)
        fitness: (pop_size,)
        pop_size, num_elite, tournament_size: ints
        crossover_rate, mutation_rate, mutation_scale: floats
        bounds_tuple: (low, high) tuple

    Returns:
        new_population: (pop_size, num_weights)
    """
    low, high = bounds_tuple
    k_sel, k_cross, k_mut = jax.random.split(key, 3)

    # Elitism: keep top individuals
    elite_idx = jnp.argsort(fitness)[-num_elite:]
    elite = population[elite_idx]

    # Number of offspring needed (round up to even for pairing)
    num_offspring = pop_size - num_elite
    num_pairs = (num_offspring + 1) // 2

    # Selection: 2 parents per pair
    sel_keys = jax.random.split(k_sel, 2)
    parent1_idx = tournament_selection(sel_keys[0], population, fitness, tournament_size)[:num_pairs]
    parent2_idx = tournament_selection(sel_keys[1], population, fitness, tournament_size)[:num_pairs]

    # Crossover — vmap over pairs, each producing (child1, child2)
    cross_keys = jax.random.split(k_cross, num_pairs)
    children1, children2 = jax.vmap(
        lambda k, p1, p2: single_point_crossover_pair(k, p1, p2, crossover_rate)
    )(cross_keys, population[parent1_idx], population[parent2_idx])

    # Interleave children: [c1_pair0, c2_pair0, c1_pair1, c2_pair1, ...]
    offspring_paired = jnp.stack([children1, children2], axis=1)
    offspring = offspring_paired.reshape(-1, population.shape[1])
    offspring = offspring[:num_offspring]  # trim if odd number

    # Mutation
    mut_keys = jax.random.split(k_mut, num_offspring)
    offspring = jax.vmap(
        lambda k, ind: gaussian_mutation(k, ind, mutation_rate, mutation_scale, low, high)
    )(mut_keys, offspring)

    return jnp.concatenate([elite, offspring], axis=0)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def create_next_generation_replacement(key, population, fitness,
                                        pop_size, num_elite, tournament_size,
                                        crossover_rate, mutation_rate,
                                        bounds_tuple):
    """Like create_next_generation but uses replacement mutation."""
    low, high = bounds_tuple
    k_sel, k_cross, k_mut = jax.random.split(key, 3)

    elite_idx = jnp.argsort(fitness)[-num_elite:]
    elite = population[elite_idx]

    num_offspring = pop_size - num_elite
    num_pairs = (num_offspring + 1) // 2

    sel_keys = jax.random.split(k_sel, 2)
    parent1_idx = tournament_selection(sel_keys[0], population, fitness, tournament_size)[:num_pairs]
    parent2_idx = tournament_selection(sel_keys[1], population, fitness, tournament_size)[:num_pairs]

    cross_keys = jax.random.split(k_cross, num_pairs)
    children1, children2 = jax.vmap(
        lambda k, p1, p2: single_point_crossover_pair(k, p1, p2, crossover_rate)
    )(cross_keys, population[parent1_idx], population[parent2_idx])

    offspring_paired = jnp.stack([children1, children2], axis=1)
    offspring = offspring_paired.reshape(-1, population.shape[1])
    offspring = offspring[:num_offspring]

    mut_keys = jax.random.split(k_mut, num_offspring)
    offspring = jax.vmap(
        lambda k, ind: replacement_mutation(k, ind, mutation_rate, low, high)
    )(mut_keys, offspring)

    return jnp.concatenate([elite, offspring], axis=0)


def initialize_population(key, pop_size, num_weights, low, high):
    """Initialize population with uniform random weights."""
    return jax.random.uniform(key, (pop_size, num_weights),
                               minval=low, maxval=high)
