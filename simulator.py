"""JAX simulation loop — single rollout using lax.scan.

One control step = sense + think + act + N physics substeps.
The full episode is a lax.scan over max_steps control steps.
Done trials continue executing but state is frozen (masked).
"""

import jax
import jax.numpy as jnp

from .types import SimState, RobotState, BallState, NNState, NoiseState, Metrics
from .kinematics import robot_substep, finalize_heading, resolve_wall_push, rate_limit_wheels
from .ball_physics import ball_substep, handle_robot_collision
from .sensors import get_ir_readings, get_vision_inputs, _normalize_sensors
from .noise import (init_noise, apply_sensor_noise, apply_motor_noise,
                    apply_vision_noise)
from .neural_network import nn_forward
from .pitch import is_goal_scored, is_ball_out_of_play


def init_state(rx, ry, rq, bx, by, key, params):
    """Initialize simulation state for one trial.

    Args:
        rx, ry, rq: robot initial pose
        bx, by: ball initial position
        key: JAX PRNGKey for this trial
        params: StaticParams
    """
    k1, k2 = jax.random.split(key)

    # Use float64 if x64 is enabled, else float32
    dtype = jnp.float64 if params.use_x64 else jnp.float32

    robot = RobotState(
        x=dtype(rx), y=dtype(ry), q=dtype(rq),
        left_actual=dtype(0.0), right_actual=dtype(0.0))

    ball = BallState(
        x=dtype(bx), y=dtype(by),
        vx=dtype(0.0), vy=dtype(0.0), omega=dtype(0.0))

    nn = NNState(h=jnp.zeros(params.hidden_size, dtype=dtype))

    noise = init_noise(k1, params)

    # Initial ball-to-goal distance (right goal for blue team)
    target_goal_x = params.half_length
    init_btg = jnp.sqrt((bx - target_goal_x) ** 2 + by ** 2)

    metrics = Metrics(
        ball_touches=jnp.int32(0),
        ball_in_contact=jnp.bool_(False),
        min_ball_to_goal=init_btg,
        initial_ball_to_goal=init_btg,
        total_distance=dtype(0.0))

    return SimState(
        robot=robot, ball=ball, nn=nn, noise=noise, metrics=metrics,
        step=jnp.int32(0),
        done=jnp.bool_(False),
        goal_scored=jnp.int32(0),
        goal_scored_step=jnp.int32(-1),
        rng_key=k2)


def step_fn(state, weights, params):
    """One control step: sense → think → act → physics substeps → check done.

    Args:
        state: SimState
        weights: (num_weights,) flat NN weights
        params: StaticParams (static, not traced)
    Returns:
        Updated SimState
    """
    # Split RNG for this step
    key, k_sensor, k_motor, k_vision = jax.random.split(state.rng_key, 4)

    robot = state.robot
    ball = state.ball
    noise = state.noise

    # Target goal is right goal (blue team)
    target_goal_x = params.half_length

    # === SENSE: IR ===
    raw_distances = get_ir_readings(robot, ball, params)

    # Apply sensor noise to raw distances (in metres)
    # Always compute both paths, select with jnp.where to avoid dtype mismatch
    noisy_distances, noise_after_sensor = apply_sensor_noise(
        raw_distances, noise, k_sensor, params)
    sensor_out = jnp.where(params.sensor_noise_enabled, noisy_distances, raw_distances)
    noise = jax.tree.map(
        lambda a, b: jnp.where(params.sensor_noise_enabled, a, b),
        noise_after_sensor, noise)

    # Clamp to non-negative, clip to sensor range, then normalize
    clipped = jnp.clip(sensor_out, params.sensor_min, params.sensor_max)
    ir_final = _normalize_sensors(clipped, params)

    # Vision
    vision_raw, ball_vis, goal_vis, ball_dist_raw, ball_angle_raw, \
        goal_dist_raw, goal_angle_raw = \
        get_vision_inputs(robot, ball, target_goal_x, params)

    # Apply vision noise — always compute, select with jnp.where
    bd, ba, gd, ga, bv_noisy, gv_noisy, noise_after_vision = apply_vision_noise(
        ball_dist_raw, ball_angle_raw,
        goal_dist_raw, goal_angle_raw,
        ball_vis, goal_vis,
        noise, k_vision, params)
    # Re-normalize after noise
    bd_n = jnp.clip(bd, 0.0, params.ball_max_range) / params.ball_max_range
    ba_n = jnp.clip(ba / jnp.pi, -1.0, 1.0)
    gd_n = jnp.clip(gd, 0.0, params.goal_max_range) / params.goal_max_range
    ga_n = jnp.clip(ga / jnp.pi, -1.0, 1.0)
    bd_n = jnp.where(bv_noisy, bd_n, 1.0)
    ba_n = jnp.where(bv_noisy, ba_n, 0.0)
    gd_n = jnp.where(gv_noisy, gd_n, 1.0)
    ga_n = jnp.where(gv_noisy, ga_n, 0.0)
    vision_noisy = jnp.array([bd_n, ba_n, gd_n, ga_n])

    vision_final = jnp.where(params.vision_noise_enabled, vision_noisy, vision_raw)
    noise = jax.tree.map(
        lambda a, b: jnp.where(params.vision_noise_enabled, a, b),
        noise_after_vision, noise)

    # Concatenate inputs: 8 IR + 4 vision
    inputs = jnp.concatenate([ir_final, vision_final])

    # === THINK ===
    (left_cmd, right_cmd), h_new = nn_forward(inputs, weights, state.nn.h, params)

    # Floreano wheel mapping: sigmoid output [0,1] → [-1,1]
    left_cmd = jnp.where(params.use_floreano_mapping, 2.0 * left_cmd - 1.0, left_cmd)
    right_cmd = jnp.where(params.use_floreano_mapping, 2.0 * right_cmd - 1.0, right_cmd)

    # === ACT ===
    # Rate-limit wheel commands
    left_actual, right_actual = rate_limit_wheels(
        left_cmd, right_cmd, robot.left_actual, robot.right_actual, params.max_cmd_delta)

    # Convert to wheel speeds
    left_speed = left_actual * params.max_wheel_speed
    right_speed = right_actual * params.max_wheel_speed

    # Apply motor noise — always compute, select with jnp.where
    noisy_left, noisy_right, noise_after_motor = apply_motor_noise(
        left_speed, right_speed, noise, k_motor, params)
    left_speed = jnp.where(params.motor_noise_enabled, noisy_left, left_speed)
    right_speed = jnp.where(params.motor_noise_enabled, noisy_right, right_speed)
    noise = jax.tree.map(
        lambda a, b: jnp.where(params.motor_noise_enabled, a, b),
        noise_after_motor, noise)

    # Save old position for distance tracking
    old_x, old_y = robot.x, robot.y

    # Update robot with new wheel actuals
    robot = robot._replace(left_actual=left_actual, right_actual=right_actual)

    # === PHYSICS SUBSTEPS ===
    def substep_body(carry, _):
        robot_s, ball_s, touches, in_contact = carry

        # Move robot
        robot_s = robot_substep(robot_s, left_speed, right_speed, params)

        # Robot-ball collision
        ball_s, touched = handle_robot_collision(
            ball_s, robot_s, left_speed, right_speed, params)

        # Count touch (edge detection: not-in-contact → in-contact)
        new_touch = touched & (~in_contact)
        touches = touches + jnp.int32(new_touch)

        # Ball substep (move + wall collision + friction)
        ball_s = ball_substep(ball_s, params)

        return (robot_s, ball_s, touches, touched), None

    init_carry = (robot, ball, state.metrics.ball_touches, state.metrics.ball_in_contact)
    (robot, ball, touches, in_contact), _ = jax.lax.scan(
        substep_body, init_carry, None, length=params.num_substeps)

    # Final separation pass
    ball, _ = handle_robot_collision(ball, robot, left_speed, right_speed, params)

    # Finalize heading
    robot = finalize_heading(robot)

    # Wall push (robot vs goal nets)
    robot = resolve_wall_push(robot, params)

    # === METRICS ===
    dx = robot.x - old_x
    dy = robot.y - old_y
    dist_moved = jnp.sqrt(dx ** 2 + dy ** 2)

    ball_to_goal = jnp.sqrt((ball.x - target_goal_x) ** 2 + ball.y ** 2)
    min_btg = jnp.minimum(state.metrics.min_ball_to_goal, ball_to_goal)

    metrics = Metrics(
        ball_touches=touches,
        ball_in_contact=in_contact,
        min_ball_to_goal=min_btg,
        initial_ball_to_goal=state.metrics.initial_ball_to_goal,
        total_distance=state.metrics.total_distance + dist_moved)

    # === CHECK DONE ===
    goal = is_goal_scored(ball.x, ball.y, params.ball_radius, params)
    scored = goal != 0
    out = is_ball_out_of_play(ball.x, ball.y, params.ball_radius, params)

    new_goal_scored = jnp.where(scored & (state.goal_scored == 0),
                                 goal, state.goal_scored)
    new_goal_step = jnp.where(scored & (state.goal_scored == 0),
                               state.step, state.goal_scored_step)
    new_done = state.done | scored | out

    new_state = SimState(
        robot=robot, ball=ball,
        nn=NNState(h=h_new),
        noise=noise, metrics=metrics,
        step=state.step + 1,
        done=new_done,
        goal_scored=new_goal_scored,
        goal_scored_step=new_goal_step,
        rng_key=key)

    return new_state


def rollout(weights, rx, ry, rq, bx, by, key, params):
    """Run a full episode. Returns final SimState.

    Args:
        weights: (num_weights,) flat NN weights
        rx, ry, rq: robot initial pose
        bx, by: ball initial position
        key: PRNGKey
        params: StaticParams
    """
    init = init_state(rx, ry, rq, bx, by, key, params)

    def scan_step(state, _):
        # Compute new state
        new_state = step_fn(state, weights, params)
        # If already done, freeze state (but increment step)
        frozen = state._replace(step=state.step + 1, rng_key=new_state.rng_key)
        out = jax.tree.map(
            lambda old, new: jnp.where(state.done, old, new),
            frozen, new_state)
        return out, None

    final_state, _ = jax.lax.scan(scan_step, init, None, length=params.max_steps)
    return final_state


def rollout_with_trajectory(weights, rx, ry, rq, bx, by, key, params):
    """Like rollout(), but also returns the full per-step trajectory.

    For visualization/debugging only — uses more memory than rollout().
    Trajectory has leading dimension (max_steps,). Use traj.done to find
    when the trial actually ended.
    """
    init = init_state(rx, ry, rq, bx, by, key, params)

    def scan_step(state, _):
        new_state = step_fn(state, weights, params)
        frozen = state._replace(step=state.step + 1, rng_key=new_state.rng_key)
        out = jax.tree.map(
            lambda old, new: jnp.where(state.done, old, new),
            frozen, new_state)
        return out, out

    final_state, trajectory = jax.lax.scan(scan_step, init, None, length=params.max_steps)
    return final_state, trajectory
