"""Differential-drive kinematics."""

import jax.numpy as jnp
from .types import RobotState


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def robot_substep(robot, left_speed, right_speed, params):
    """One physics substep of differential-drive kinematics.

    Args:
        robot: RobotState
        left_speed, right_speed: wheel speeds (after noise, rate-limiting)
        params: StaticParams
    Returns:
        Updated RobotState
    """
    R = params.wheel_radius
    L = params.half_wheelbase
    dt = params.physics_timestep

    v = R * (left_speed + right_speed) / 2.0
    omega = R * (right_speed - left_speed) / (2.0 * L)

    x = robot.x + jnp.cos(robot.q) * v * dt
    y = robot.y + jnp.sin(robot.q) * v * dt
    q = robot.q + omega * dt

    return robot._replace(x=x, y=y, q=q)


def finalize_heading(robot):
    """Wrap heading to [-pi, pi] after all substeps."""
    return robot._replace(q=wrap_to_pi(robot.q))


def resolve_wall_push(robot, params):
    """Push robot out of goal-net walls (push mode only).

    Only checks distance to goal segments since field boundary is open.
    """
    from .pitch import distance_to_segments

    wall_dist = distance_to_segments(robot.x, robot.y, params)
    overlap = params.body_radius - wall_dist
    needs_push = overlap > 0.0

    # Compute push direction (away from nearest segment point)
    fx = robot.x - params.seg_x1
    fy = robot.y - params.seg_y1
    len_sq = jnp.clip(params.seg_len_sq, 1e-12, None)
    t = jnp.clip((fx * params.seg_dx + fy * params.seg_dy) / len_sq, 0.0, 1.0)
    proj_x = params.seg_x1 + t * params.seg_dx
    proj_y = params.seg_y1 + t * params.seg_dy
    dists = jnp.sqrt((robot.x - proj_x) ** 2 + (robot.y - proj_y) ** 2)

    i = jnp.argmin(dists)
    dist = dists[i]
    safe_dist = jnp.maximum(dist, 1e-12)
    nx = (robot.x - proj_x[i]) / safe_dist
    ny = (robot.y - proj_y[i]) / safe_dist

    push_x = jnp.where(needs_push, robot.x + nx * overlap, robot.x)
    push_y = jnp.where(needs_push, robot.y + ny * overlap, robot.y)

    return robot._replace(x=push_x, y=push_y)


def rate_limit_wheels(left_cmd, right_cmd, left_actual, right_actual, max_delta):
    """Apply rate limiting to wheel commands.

    Returns updated (left_actual, right_actual).
    """
    left_cmd = jnp.clip(left_cmd, -1.0, 1.0)
    right_cmd = jnp.clip(right_cmd, -1.0, 1.0)

    dl = jnp.clip(left_cmd - left_actual, -max_delta, max_delta)
    dr = jnp.clip(right_cmd - right_actual, -max_delta, max_delta)

    return left_actual + dl, right_actual + dr
