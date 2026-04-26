"""Vectorized population evaluator using vmap.

Evaluates all individuals × all trials in a single vectorized call.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .simulator import rollout
from .fitness import penalty_sparse


def make_evaluate_population(params):
    """Create a JIT-compiled function that evaluates an entire population.

    Args:
        params: StaticParams (treated as static for JIT)

    Returns:
        evaluate_fn(population, init_states, keys) -> fitness_array
    """

    @partial(jax.jit, static_argnums=())
    def evaluate_population(population, init_states, keys):
        """Evaluate all individuals on all trials.

        Args:
            population: (pop_size, num_weights) weight arrays
            init_states: (pop_size, num_trials, 5) initial (rx, ry, rq, bx, by)
            keys: (pop_size, num_trials, 2) PRNGKeys

        Returns:
            fitness: (pop_size,) mean fitness over trials
            goal_rates: (pop_size,) fraction of trials with goals
        """
        # vmap over individuals, then over trials
        def eval_one_trial(weights, init, key):
            rx, ry, rq, bx, by = init[0], init[1], init[2], init[3], init[4]
            final = rollout(weights, rx, ry, rq, bx, by, key, params)
            fit = penalty_sparse(final.goal_scored, final.goal_scored_step,
                                 params.max_steps)
            return fit, final.goal_scored

        def eval_one_individual(weights, inits, keys_i):
            # vmap over trials
            fits, goals = jax.vmap(
                lambda init, key: eval_one_trial(weights, init, key)
            )(inits, keys_i)
            return jnp.mean(fits), jnp.mean(goals == 1)  # 1 = right goal = target

        # vmap over population
        fitness, goal_rates = jax.vmap(eval_one_individual)(
            population, init_states, keys)

        return fitness, goal_rates

    return evaluate_population
