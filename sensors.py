"""Sensor models — IR ray casting and vision system."""

import jax.numpy as jnp
from .pitch import ray_cast_multi
from .kinematics import wrap_to_pi


def _ray_circle_distance(ox, oy, angle, cx, cy, radius):
    """Distance along ray to intersection with circle. Returns inf if no hit."""
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)
    fx = ox - cx
    fy = oy - cy

    a = dx * dx + dy * dy  # always 1.0 for unit direction
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4.0 * a * c
    has_solution = discriminant >= 0.0

    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Take nearest positive hit
    t1_valid = t1 > 1e-6
    t2_valid = t2 > 1e-6
    t = jnp.where(t1_valid, t1, jnp.where(t2_valid, t2, jnp.inf))
    t = jnp.where(has_solution, t, jnp.inf)
    return t


def get_ir_readings(robot, ball, params):
    """Compute 8 IR sensor raw distances in metres.

    Rays are cast against goal segments and the ball.
    Returns (8,) raw distances — no clipping, no normalization.
    """
    directions = params.sensor_angles + robot.q  # world-frame angles

    # Cast against goal segments
    wall_dists = ray_cast_multi(robot.x, robot.y, directions, params)

    # Cast against ball
    ball_dists = _ray_circle_distances_batch(
        robot.x, robot.y, directions, ball.x, ball.y, params.ball_radius)

    # Take minimum
    return jnp.minimum(wall_dists, ball_dists)


def _ray_circle_distances_batch(ox, oy, directions, cx, cy, radius):
    """Batch ray-circle intersection for all sensor rays."""
    dx = jnp.cos(directions)
    dy = jnp.sin(directions)
    fx = ox - cx
    fy = oy - cy

    # a = 1.0 for unit directions
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4.0 * c
    has_solution = discriminant >= 0.0

    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    t1_valid = t1 > 1e-6
    t2_valid = t2 > 1e-6
    t = jnp.where(t1_valid, t1, jnp.where(t2_valid, t2, jnp.inf))
    return jnp.where(has_solution, t, jnp.inf)


def _normalize_sensors(distances, params):
    """Normalize sensor distances based on configured method."""
    s_min = params.sensor_min
    s_max = params.sensor_max
    rng = s_max - s_min

    # minmax: (d - min) / (max - min) → 0=close, 1=far
    minmax = (distances - s_min) / jnp.maximum(rng, 1e-12)

    # inverse: 1/d normalized
    inv = 1.0 / jnp.maximum(distances, 1e-6)
    inv_min = 1.0 / s_max
    inv_max = 1.0 / jnp.maximum(s_min, 1e-6)
    inverse = (inv - inv_min) / jnp.maximum(inv_max - inv_min, 1e-12)

    # linear_inverted: 1 - minmax → 0=far, 1=close
    lin_inv = 1.0 - minmax

    # Select based on config (0=minmax, 1=inverse, 2=linear_inverted)
    norm = params.sensor_normalization
    result = jnp.where(norm == 0, minmax,
             jnp.where(norm == 1, inverse, lin_inv))
    return result


def get_vision_inputs(robot, ball, target_goal_x, params):
    """Compute 4 vision inputs: ball_dist, ball_angle, goal_dist, goal_angle.

    Returns:
        vision_normalized: (4,) array — normalized, with sentinels where invisible
        ball_visible, goal_visible: bool flags
        ball_dist, ball_angle, goal_dist, goal_angle: raw values for noise injection
    """
    # Ball distance and angle
    bdx = ball.x - robot.x
    bdy = ball.y - robot.y
    ball_dist = jnp.sqrt(bdx ** 2 + bdy ** 2)
    ball_angle = wrap_to_pi(jnp.arctan2(bdy, bdx) - robot.q)

    # Ball visibility (frontal FOV check)
    ball_half_span = jnp.arcsin(jnp.clip(
        params.ball_radius / jnp.maximum(ball_dist, 1e-6), 0.0, 1.0))
    ball_in_range = ball_dist <= params.ball_max_range
    ball_in_fov = jnp.abs(ball_angle) <= params.vision_half_fov + ball_half_span
    ball_visible = ball_in_range & ball_in_fov

    # Goal distance and angle (to center of target goal line)
    gdx = target_goal_x - robot.x
    gdy = 0.0 - robot.y  # goal center is at y=0
    goal_dist = jnp.sqrt(gdx ** 2 + gdy ** 2)
    goal_angle = wrap_to_pi(jnp.arctan2(gdy, gdx) - robot.q)

    # Goal visibility (wider span due to goal width)
    goal_half_span = jnp.arctan2(params.half_goal_width,
                                  jnp.maximum(jnp.abs(gdx), 1e-6))
    goal_in_range = goal_dist <= params.goal_max_range
    goal_in_fov = jnp.abs(goal_angle) <= params.vision_half_fov + goal_half_span
    goal_visible = goal_in_range & goal_in_fov

    # Normalize
    ball_dist_norm = jnp.clip(ball_dist, 0.0, params.ball_max_range) / params.ball_max_range
    ball_angle_norm = jnp.clip(ball_angle / jnp.pi, -1.0, 1.0)
    goal_dist_norm = jnp.clip(goal_dist, 0.0, params.goal_max_range) / params.goal_max_range
    goal_angle_norm = jnp.clip(goal_angle / jnp.pi, -1.0, 1.0)

    # If not visible, return max dist and zero angle
    ball_dist_norm = jnp.where(ball_visible, ball_dist_norm, 1.0)
    ball_angle_norm = jnp.where(ball_visible, ball_angle_norm, 0.0)
    goal_dist_norm = jnp.where(goal_visible, goal_dist_norm, 1.0)
    goal_angle_norm = jnp.where(goal_visible, goal_angle_norm, 0.0)

    vision = jnp.array([ball_dist_norm, ball_angle_norm,
                         goal_dist_norm, goal_angle_norm])

    return vision, ball_visible, goal_visible, ball_dist, ball_angle, goal_dist, goal_angle
