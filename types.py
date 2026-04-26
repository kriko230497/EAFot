"""State types for the JAX simulation.

All mutable state is stored in NamedTuples of JAX arrays.
Static configuration is stored in StaticParams (also a NamedTuple).
"""

from typing import NamedTuple
import jax.numpy as jnp


class RobotState(NamedTuple):
    x: jnp.ndarray          # scalar
    y: jnp.ndarray
    q: jnp.ndarray          # heading
    left_actual: jnp.ndarray   # rate-limited wheel command
    right_actual: jnp.ndarray


class BallState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    omega: jnp.ndarray       # spin (vertical axis)


class NNState(NamedTuple):
    h: jnp.ndarray           # (hidden_size,) — zeros for feedforward


class NoiseState(NamedTuple):
    # IR sensors (8 channels)
    sensor_bias: jnp.ndarray       # (8,) fixed per trial
    sensor_ar: jnp.ndarray         # (8,) AR(1) running state
    # Motor (2 channels)
    motor_bias: jnp.ndarray        # (2,) fixed per trial
    motor_ar: jnp.ndarray          # (2,) AR(1) running state
    # Vision (4 AR(1) channels + 4 biases)
    vision_ar: jnp.ndarray         # (4,) ball_dist, ball_angle, goal_dist, goal_angle
    vision_bias: jnp.ndarray       # (4,) fixed per trial


class Metrics(NamedTuple):
    ball_touches: jnp.ndarray         # int
    ball_in_contact: jnp.ndarray      # bool — for edge detection
    min_ball_to_goal: jnp.ndarray     # float
    initial_ball_to_goal: jnp.ndarray # float
    total_distance: jnp.ndarray       # float


class SimState(NamedTuple):
    robot: RobotState
    ball: BallState
    nn: NNState
    noise: NoiseState
    metrics: Metrics
    step: jnp.ndarray             # int — current step
    done: jnp.ndarray             # bool — trial finished
    goal_scored: jnp.ndarray      # int: 0=none, 1=right, -1=left
    goal_scored_step: jnp.ndarray # int
    rng_key: jnp.ndarray          # PRNGKey


class StaticParams(NamedTuple):
    # Robot
    wheel_radius: float
    half_wheelbase: float
    body_radius: float
    robot_mass: float
    max_wheel_speed: float
    max_cmd_delta: float
    # Ball
    ball_radius: float
    ball_mass: float
    ball_inertia: float          # I = inertia_factor * m * r^2
    translational_decel: float   # rolling_friction * gravity
    wall_restitution: float
    robot_restitution: float
    wall_friction: float
    robot_friction: float
    spin_damping: float
    physics_timestep: float
    # Pitch
    half_length: float
    half_width: float
    half_goal_width: float
    goal_depth: float
    # Pitch segments (only goal segments for open mode)
    seg_x1: jnp.ndarray    # (N,)
    seg_y1: jnp.ndarray
    seg_dx: jnp.ndarray
    seg_dy: jnp.ndarray
    seg_len_sq: jnp.ndarray
    num_segments: int
    # IR Sensors
    sensor_angles: jnp.ndarray  # (8,)
    sensor_max: float
    sensor_min: float
    # Vision
    vision_half_fov: float
    ball_max_range: float
    goal_max_range: float
    # Noise params — sensors
    sensor_noise_enabled: bool
    sensor_relative_std: float
    sensor_bias_std: float
    sensor_rho: float
    # Noise params — motor
    motor_noise_enabled: bool
    motor_relative_std: float
    motor_bias_std: float
    motor_rho: float
    motor_gain_clip_lo: float
    motor_gain_clip_hi: float
    # Noise params — vision
    vision_noise_enabled: bool
    vision_rho: float
    ball_dist_bias_std: float
    ball_angle_bias_std: float
    goal_dist_bias_std: float
    goal_angle_bias_std: float
    ball_dist_std_far: float
    ball_dist_std_near: float
    ball_angle_std_far: float
    ball_angle_std_near: float
    goal_dist_std: float
    goal_angle_std: float
    near_start: float
    near_full: float
    ball_dropout_far: float
    ball_dropout_near: float
    goal_dropout_far: float
    goal_dropout_near: float
    # Simulation
    max_steps: int
    num_substeps: int
    # NN
    hidden_size: int
    input_size: int
    output_size: int
    is_elman: bool
    use_sigmoid: bool          # True=sigmoid, False=tanh
    use_floreano_mapping: bool # True=floreano wheel mapping
    # Normalization
    sensor_normalization: int  # 0=minmax, 1=inverse, 2=linear_inverted
    # Precision
    use_x64: bool              # True=float64 (NumPy parity), False=float32 (faster)
    # Validation
    validate_in_jax: bool      # True=JAX validation, False=NumPy validation
