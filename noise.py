"""AR(1) and white noise models for sensors, motors, and vision."""

import jax
import jax.numpy as jnp
from .types import NoiseState


def init_noise(key, params):
    """Initialize noise state for one trial.

    Biases are sampled once per trial (fixed). AR(1) states are sampled
    from the stationary distribution N(0, sigma^2) so the process is
    stationary from t=0.
    """
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    dtype = jnp.float64 if params.use_x64 else jnp.float32

    # Sensor biases (8 channels)
    sensor_bias = jnp.where(
        params.sensor_noise_enabled,
        jax.random.normal(k1, (8,), dtype=dtype) * params.sensor_bias_std,
        jnp.zeros(8, dtype=dtype))

    # Sensor AR(1) initial state — stationary draw
    sensor_ar = jnp.where(
        params.sensor_noise_enabled,
        jax.random.normal(k2, (8,), dtype=dtype) * params.sensor_relative_std,
        jnp.zeros(8, dtype=dtype))

    # Motor biases (2 channels)
    motor_bias = jnp.where(
        params.motor_noise_enabled,
        jax.random.normal(k3, (2,), dtype=dtype) * params.motor_bias_std,
        jnp.zeros(2, dtype=dtype))

    # Motor AR(1) initial state — stationary draw
    motor_ar = jnp.where(
        params.motor_noise_enabled,
        jax.random.normal(k4, (2,), dtype=dtype) * params.motor_relative_std,
        jnp.zeros(2, dtype=dtype))

    # Vision biases (4 channels)
    vision_bias_stds = jnp.array([
        params.ball_dist_bias_std,
        params.ball_angle_bias_std,
        params.goal_dist_bias_std,
        params.goal_angle_bias_std,
    ], dtype=dtype)
    vision_bias = jnp.where(
        params.vision_noise_enabled,
        jax.random.normal(k5, (4,), dtype=dtype) * vision_bias_stds,
        jnp.zeros(4, dtype=dtype))

    # Vision AR(1) initial state — stationary draw using far-range stds
    vision_init_stds = jnp.array([
        params.ball_dist_std_far,
        params.ball_angle_std_far,
        params.goal_dist_std,
        params.goal_angle_std,
    ], dtype=dtype)
    vision_ar = jnp.where(
        params.vision_noise_enabled,
        jax.random.normal(k6, (4,), dtype=dtype) * vision_init_stds,
        jnp.zeros(4, dtype=dtype))

    return NoiseState(
        sensor_bias=sensor_bias,
        sensor_ar=sensor_ar,
        motor_bias=motor_bias,
        motor_ar=motor_ar,
        vision_ar=vision_ar,
        vision_bias=vision_bias,
    )


def update_ar1(state, key, target_std, rho):
    """Update AR(1) process: state = rho * state + innovation.

    Returns updated state (same shape as input).
    """
    innovation_std = target_std * jnp.sqrt(1.0 - rho ** 2)
    innovation = jax.random.normal(key, state.shape) * innovation_std
    return rho * state + innovation


def apply_sensor_noise(distances, noise_state, key, params):
    """Apply correlated noise to IR sensor readings.

    Returns (noisy_distances, updated_noise_state).
    """
    # Update AR(1) sensor noise
    new_ar = update_ar1(noise_state.sensor_ar, key,
                        params.sensor_relative_std, params.sensor_rho)

    # Multiplicative gain: distance * (1 + bias + ar_state)
    gain = 1.0 + noise_state.sensor_bias + new_ar
    noisy = distances * gain

    # Clamp to non-negative
    noisy = jnp.maximum(noisy, 0.0)

    new_noise = noise_state._replace(sensor_ar=new_ar)
    return noisy, new_noise


def apply_motor_noise(left_speed, right_speed, noise_state, key, params):
    """Apply correlated noise to motor commands.

    Returns (noisy_left, noisy_right, updated_noise_state).
    """
    new_ar = update_ar1(noise_state.motor_ar, key,
                        params.motor_relative_std, params.motor_rho)

    gains = 1.0 + noise_state.motor_bias + new_ar
    gains = jnp.clip(gains, params.motor_gain_clip_lo, params.motor_gain_clip_hi)

    speeds = jnp.array([left_speed, right_speed])
    noisy_speeds = speeds * gains

    new_noise = noise_state._replace(motor_ar=new_ar)
    return noisy_speeds[0], noisy_speeds[1], new_noise


def close_range_alpha(distance, params):
    """Compute close-range interpolation factor.

    0 when distance >= near_start, 1 when distance <= near_full.
    """
    alpha = (params.near_start - distance) / jnp.maximum(
        params.near_start - params.near_full, 1e-12)
    return jnp.clip(alpha, 0.0, 1.0)


def apply_vision_noise(ball_dist, ball_angle, goal_dist, goal_angle,
                       ball_visible, goal_visible,
                       noise_state, key, params):
    """Apply correlated noise to vision measurements.

    Returns (noisy_ball_dist, noisy_ball_angle, noisy_goal_dist, noisy_goal_angle,
             ball_visible, goal_visible, updated_noise_state).
    """
    k1, k2 = jax.random.split(key)

    # Close-range alpha for ball
    alpha_ball = close_range_alpha(ball_dist, params)

    # Target stds (interpolated for ball)
    target_stds = jnp.array([
        (1.0 - alpha_ball) * params.ball_dist_std_far + alpha_ball * params.ball_dist_std_near,
        (1.0 - alpha_ball) * params.ball_angle_std_far + alpha_ball * params.ball_angle_std_near,
        params.goal_dist_std,
        params.goal_angle_std,
    ])

    # Update AR(1)
    new_ar = update_ar1(noise_state.vision_ar, k1, target_stds, params.vision_rho)

    # Apply noise + bias
    noisy_ball_dist = ball_dist + noise_state.vision_bias[0] + new_ar[0]
    noisy_ball_angle = ball_angle + noise_state.vision_bias[1] + new_ar[1]
    noisy_goal_dist = goal_dist + noise_state.vision_bias[2] + new_ar[2]
    noisy_goal_angle = goal_angle + noise_state.vision_bias[3] + new_ar[3]

    # Clamp distances
    noisy_ball_dist = jnp.clip(noisy_ball_dist, 0.0, params.ball_max_range)
    noisy_goal_dist = jnp.clip(noisy_goal_dist, 0.0, params.goal_max_range)

    # Wrap angles
    noisy_ball_angle = (noisy_ball_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    noisy_goal_angle = (noisy_goal_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # Dropout
    ball_p_dropout = ((1.0 - alpha_ball) * params.ball_dropout_far +
                      alpha_ball * params.ball_dropout_near)
    alpha_goal = close_range_alpha(goal_dist, params)
    goal_p_dropout = ((1.0 - alpha_goal) * params.goal_dropout_far +
                      alpha_goal * params.goal_dropout_near)

    dropout_samples = jax.random.uniform(k2, (2,))
    ball_dropped = dropout_samples[0] < ball_p_dropout
    goal_dropped = dropout_samples[1] < goal_p_dropout

    ball_visible = ball_visible & (~ball_dropped)
    goal_visible = goal_visible & (~goal_dropped)

    # If not visible, return max range and zero angle
    noisy_ball_dist = jnp.where(ball_visible, noisy_ball_dist, params.ball_max_range)
    noisy_ball_angle = jnp.where(ball_visible, noisy_ball_angle, 0.0)
    noisy_goal_dist = jnp.where(goal_visible, noisy_goal_dist, params.goal_max_range)
    noisy_goal_angle = jnp.where(goal_visible, noisy_goal_angle, 0.0)

    new_noise = noise_state._replace(vision_ar=new_ar)
    return (noisy_ball_dist, noisy_ball_angle,
            noisy_goal_dist, noisy_goal_angle,
            ball_visible, goal_visible, new_noise)
