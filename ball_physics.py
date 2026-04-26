"""Ball physics — dynamics, wall collision, robot collision.

All functions are branch-free (using jnp.where) for JAX compatibility.
Only goal segments are solid (open mode).
"""

import jax.numpy as jnp
from .types import BallState


def ball_substep(ball, params):
    """One physics substep: move, wall collision, friction, spin damping.

    Args:
        ball: BallState
        params: StaticParams
    Returns:
        Updated BallState
    """
    dt = params.physics_timestep

    # 1. Position update
    x = ball.x + ball.vx * dt
    y = ball.y + ball.vy * dt

    # 2. Wall collision (goal segments only)
    x, y, vx, vy, omega = _handle_wall_collisions(
        x, y, ball.vx, ball.vy, ball.omega, params)

    # 3. Rolling friction
    speed_sq = vx ** 2 + vy ** 2
    speed = jnp.sqrt(jnp.maximum(speed_sq, 0.0))
    decel = params.translational_decel * dt
    # If decel >= speed, stop completely; else reduce proportionally
    stopped = (speed_sq <= 1e-10) | (decel >= speed)
    factor = jnp.where(stopped, 0.0, (speed - decel) / jnp.maximum(speed, 1e-12))
    vx = vx * factor
    vy = vy * factor

    # 4. Spin damping
    sd = params.spin_damping
    spin_factor = jnp.where(
        (sd > 0.0) & (jnp.abs(omega) >= 1e-6),
        jnp.maximum(0.0, 1.0 - sd * dt),
        jnp.where(jnp.abs(omega) < 1e-6, 0.0, 1.0)
    )
    omega = omega * spin_factor

    return BallState(x=x, y=y, vx=vx, vy=vy, omega=omega)


def _handle_wall_collisions(x, y, vx, vy, omega, params):
    """Handle ball collision with goal-net segments.

    Finds deepest overlap, applies normal impulse + tangential friction.
    """
    r = params.ball_radius
    mass = params.ball_mass
    inertia = params.ball_inertia

    # Quick exit: if ball is far from any goal area
    # (still compute everything but mask with jnp.where)

    # Project ball center onto each segment
    fx = x - params.seg_x1
    fy = y - params.seg_y1
    len_sq = jnp.clip(params.seg_len_sq, 1e-12, None)
    t = jnp.clip((fx * params.seg_dx + fy * params.seg_dy) / len_sq, 0.0, 1.0)

    proj_x = params.seg_x1 + t * params.seg_dx
    proj_y = params.seg_y1 + t * params.seg_dy

    dx = x - proj_x
    dy = y - proj_y
    dists = jnp.sqrt(dx ** 2 + dy ** 2)

    # Overlap per segment
    overlaps = r - dists

    # Find deepest overlap
    i = jnp.argmax(overlaps)
    max_overlap = overlaps[i]
    has_collision = max_overlap > 0.0

    # Normal direction (away from wall)
    dist_i = jnp.maximum(dists[i], 1e-12)
    nx = dx[i] / dist_i
    ny = dy[i] / dist_i

    # Push ball out of wall
    x_new = x + nx * max_overlap
    y_new = y + ny * max_overlap

    # Tangent direction
    tx = -ny
    ty = nx

    # Contact velocity (ball surface, including spin)
    ball_cvx = vx + omega * r * ny
    ball_cvy = vy - omega * r * nx

    v_rel_n = ball_cvx * nx + ball_cvy * ny
    v_rel_t = ball_cvx * tx + ball_cvy * ty

    # Normal impulse (only if approaching)
    approaching = v_rel_n < 0.0
    apply_impulse = has_collision & approaching
    jn = -(1.0 + params.wall_restitution) * v_rel_n * mass

    vx_new = vx + jnp.where(apply_impulse, (jn / mass) * nx, 0.0)
    vy_new = vy + jnp.where(apply_impulse, (jn / mass) * ny, 0.0)

    # Tangential friction impulse
    denom_t = 1.0 / mass + r ** 2 / inertia
    jt_unc = -v_rel_t / denom_t
    jt_max = params.wall_friction * jnp.abs(jn)
    jt = jnp.clip(jt_unc, -jt_max, jt_max)
    has_friction = apply_impulse & (jnp.abs(jt) > 1e-12)

    vx_new = vx_new + jnp.where(has_friction, (jt / mass) * tx, 0.0)
    vy_new = vy_new + jnp.where(has_friction, (jt / mass) * ty, 0.0)
    omega_new = omega + jnp.where(has_friction, (-jt * r) / inertia, 0.0)

    # Apply conditionally
    x_out = jnp.where(has_collision, x_new, x)
    y_out = jnp.where(has_collision, y_new, y)
    vx_out = jnp.where(has_collision, vx_new, vx)
    vy_out = jnp.where(has_collision, vy_new, vy)
    omega_out = jnp.where(has_collision, omega_new, omega)

    return x_out, y_out, vx_out, vy_out, omega_out


def handle_robot_collision(ball, robot, left_speed, right_speed, params):
    """Handle ball-robot collision. Returns (new_ball, touched).

    Robot is kinematic (not affected). Ball receives impulse.
    """
    r_ball = params.ball_radius
    r_robot = params.body_radius
    min_dist = r_robot + r_ball

    dx = ball.x - robot.x
    dy = ball.y - robot.y
    dist_sq = dx ** 2 + dy ** 2
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-24))

    has_collision = dist_sq < min_dist ** 2

    # Normal direction (from robot to ball)
    # If dist ~ 0, use robot heading as fallback
    safe_dist = jnp.maximum(dist, 1e-12)
    nx_dist = dx / safe_dist
    ny_dist = dy / safe_dist
    nx_heading = jnp.cos(robot.q)
    ny_heading = jnp.sin(robot.q)
    use_heading = dist < 1e-12
    nx = jnp.where(use_heading, nx_heading, nx_dist)
    ny = jnp.where(use_heading, ny_heading, ny_dist)

    tx = -ny
    ty = nx

    # Position correction
    overlap = jnp.maximum(0.0, min_dist - dist)
    bx_new = ball.x + nx * overlap
    by_new = ball.y + ny * overlap

    # Robot contact velocity
    R = params.wheel_radius
    L = params.half_wheelbase
    v_linear = R * (left_speed + right_speed) / 2.0
    omega_robot = R * (right_speed - left_speed) / (2.0 * L)
    contact_vx = jnp.cos(robot.q) * v_linear - omega_robot * r_robot * ny
    contact_vy = jnp.sin(robot.q) * v_linear + omega_robot * r_robot * nx

    # Ball surface velocity (including spin)
    ball_cvx = ball.vx + ball.omega * r_ball * ny
    ball_cvy = ball.vy - ball.omega * r_ball * nx

    # Relative velocity
    rel_vx = ball_cvx - contact_vx
    rel_vy = ball_cvy - contact_vy
    v_rel_n = rel_vx * nx + rel_vy * ny
    v_rel_t = rel_vx * tx + rel_vy * ty

    # Only apply if approaching
    approaching = v_rel_n < 0.0
    apply = has_collision & approaching

    # Finite-mass scale
    mass_factor = params.robot_mass / (params.robot_mass + params.ball_mass)

    # Normal impulse
    denom_n = 1.0 / params.ball_mass
    jn = -(1.0 + params.robot_restitution) * v_rel_n / denom_n * mass_factor

    vx_new = ball.vx + jnp.where(apply, (jn / params.ball_mass) * nx, 0.0)
    vy_new = ball.vy + jnp.where(apply, (jn / params.ball_mass) * ny, 0.0)

    # Tangential friction + spin
    inertia = params.ball_inertia
    denom_t = 1.0 / params.ball_mass + r_ball ** 2 / inertia
    jt_unc = -v_rel_t / denom_t
    jt_max = params.robot_friction * jnp.abs(jn)
    jt = jnp.clip(jt_unc, -jt_max, jt_max) * mass_factor
    has_friction = apply & (jnp.abs(jt) > 1e-12)

    vx_new = vx_new + jnp.where(has_friction, (jt / params.ball_mass) * tx, 0.0)
    vy_new = vy_new + jnp.where(has_friction, (jt / params.ball_mass) * ty, 0.0)
    omega_new = ball.omega + jnp.where(has_friction,
                                        (-jt * r_ball) / inertia, 0.0)

    # Apply conditionally
    new_ball = BallState(
        x=jnp.where(has_collision, bx_new, ball.x),
        y=jnp.where(has_collision, by_new, ball.y),
        vx=jnp.where(has_collision, vx_new, ball.vx),
        vy=jnp.where(has_collision, vy_new, ball.vy),
        omega=jnp.where(has_collision, omega_new, ball.omega),
    )

    return new_ball, has_collision
