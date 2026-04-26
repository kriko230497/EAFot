"""Fitness functions."""

import jax.numpy as jnp


def penalty_sparse(goal_scored, goal_scored_step, max_steps, target_side=1):
    """Sparse penalty kick fitness.

    +1 if goal scored in the correct goal (target_side).
    Time bonus: multiply by (1 - 0.5 * step_fraction) — removed per config.
    Returns 0.0 or 1.0.
    """
    correct = goal_scored == target_side
    return jnp.where(correct, 1.0, 0.0)
