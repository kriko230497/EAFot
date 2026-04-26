"""Load YAML config and convert to JAX-compatible StaticParams."""

import yaml
import math
import jax
import jax.numpy as jnp
from .types import StaticParams
from .pitch import build_goal_segments


def load_config(yaml_path: str) -> StaticParams:
    """Load YAML config and produce a frozen StaticParams for JIT."""
    with open(yaml_path) as f:
        c = yaml.safe_load(f)

    sim = c.get('simulation', {})
    robot = c.get('robot', {})
    ball = c.get('ball', {})
    pitch = c.get('pitch', {})
    sensors = c.get('sensors', {})
    vision = c.get('vision', {})
    nn = c.get('neural_network', {})
    motor = c.get('motor', {})

    # JAX settings
    jax_cfg = c.get('jax', {})
    use_x64 = jax_cfg.get('use_x64', False)
    validate_in_jax = jax_cfg.get('validate_in_jax', False)
    if use_x64:
        jax.config.update("jax_enable_x64", True)

    # Pitch geometry
    hl = pitch['length'] / 2.0
    hw = pitch['width'] / 2.0
    hgw = pitch['goal_width'] / 2.0
    gd = pitch['goal_depth']

    # Build goal segments (open mode — only goals are solid)
    seg_x1, seg_y1, seg_dx, seg_dy, seg_len_sq = build_goal_segments(
        hl, hw, hgw, gd)

    # Physics
    timestep = sim.get('timestep', 0.1)
    physics_timestep = sim.get('physics_timestep', 0.01)
    num_substeps = round(timestep / physics_timestep)

    ball_r = ball.get('radius', 0.0335)
    ball_m = ball.get('mass', 0.0577)
    inertia_factor = ball.get('inertia_factor', 0.6667)
    ball_inertia = inertia_factor * ball_m * ball_r ** 2
    rolling_friction = ball.get('rolling_friction', 0.04)
    gravity = ball.get('gravity', 9.81)

    # Sensor noise
    sn = sensors.get('noise', {})
    sensor_noise_enabled = sn.get('enabled', False)

    # Motor noise
    mn = motor.get('noise', {})
    motor_noise_enabled = mn.get('enabled', False)
    gain_clip = mn.get('gain_clip', [0.75, 1.25])

    # Vision noise
    vn = vision.get('noise', {})
    vision_noise_enabled = vn.get('enabled', False)

    # NN
    hidden_size = nn.get('hidden_size', 5)
    vision_inputs = nn.get('vision_inputs', 4)
    input_size = len(sensors.get('angles', list(range(8)))) + vision_inputs
    output_size = nn.get('output_size', 2)
    is_elman = nn.get('type', 'feedforward') == 'elman'
    activation = nn.get('activation', 'tanh')
    wheel_mapping = nn.get('wheel_output_mapping', 'symmetric')

    # Sensor normalization
    norm_str = sensors.get('normalization', 'minmax')
    norm_map = {'minmax': 0, 'inverse': 1, 'linear_inverted': 2}
    sensor_normalization = norm_map.get(norm_str, 0)

    return StaticParams(
        # Robot
        wheel_radius=robot.get('wheel_radius', 0.021),
        half_wheelbase=robot.get('half_wheelbase', 0.0527),
        body_radius=robot.get('body_radius', 0.0704),
        robot_mass=robot.get('mass', 0.566),
        max_wheel_speed=robot.get('max_wheel_speed', 14.3),
        max_cmd_delta=robot.get('max_wheel_cmd_delta', 0.15),
        # Ball
        ball_radius=ball_r,
        ball_mass=ball_m,
        ball_inertia=ball_inertia,
        translational_decel=rolling_friction * gravity,
        wall_restitution=ball.get('wall_restitution', 0.745),
        robot_restitution=ball.get('robot_restitution', 0.5),
        wall_friction=ball.get('wall_friction', 0.23),
        robot_friction=ball.get('robot_friction', 0.23),
        spin_damping=ball.get('spin_damping', 0.0),
        physics_timestep=physics_timestep,
        # Pitch
        half_length=hl,
        half_width=hw,
        half_goal_width=hgw,
        goal_depth=gd,
        seg_x1=jnp.array(seg_x1, dtype=jnp.float32),
        seg_y1=jnp.array(seg_y1, dtype=jnp.float32),
        seg_dx=jnp.array(seg_dx, dtype=jnp.float32),
        seg_dy=jnp.array(seg_dy, dtype=jnp.float32),
        seg_len_sq=jnp.array(seg_len_sq, dtype=jnp.float32),
        num_segments=len(seg_x1),
        # Sensors
        sensor_angles=jnp.array(sensors.get('angles', [
            -2.53, -1.571, -0.785, -0.175, 0.175, 0.785, 1.571, 2.53
        ]), dtype=jnp.float32),
        sensor_max=sensors.get('max_range', 0.25),
        sensor_min=sensors.get('min_range', 0.005),
        # Vision
        vision_half_fov=math.radians(vision.get('horizontal_fov_deg', 131.0)) / 2.0,
        ball_max_range=vision.get('ball_max_range', 1.9),
        goal_max_range=vision.get('goal_max_range', 3.6),
        # Sensor noise
        sensor_noise_enabled=sensor_noise_enabled,
        sensor_relative_std=sn.get('relative_std', 0.04),
        sensor_bias_std=sn.get('bias_std', 0.015),
        sensor_rho=sn.get('rho', 0.92),
        # Motor noise
        motor_noise_enabled=motor_noise_enabled,
        motor_relative_std=mn.get('relative_std', 0.03),
        motor_bias_std=mn.get('bias_std', 0.02),
        motor_rho=mn.get('rho', 0.95),
        motor_gain_clip_lo=gain_clip[0] if gain_clip else 0.75,
        motor_gain_clip_hi=gain_clip[1] if gain_clip else 1.25,
        # Vision noise
        vision_noise_enabled=vision_noise_enabled,
        vision_rho=vn.get('rho', 0.95),
        ball_dist_bias_std=vn.get('ball_dist_bias_std_m', 0.015),
        ball_angle_bias_std=vn.get('ball_angle_bias_std_rad', 0.01),
        goal_dist_bias_std=vn.get('goal_dist_bias_std_m', 0.02),
        goal_angle_bias_std=vn.get('goal_angle_bias_std_rad', 0.006),
        ball_dist_std_far=vn.get('ball_dist_std_far_m', 0.1),
        ball_dist_std_near=vn.get('ball_dist_std_near_m', 0.16),
        ball_angle_std_far=vn.get('ball_angle_std_far_rad', 0.015),
        ball_angle_std_near=vn.get('ball_angle_std_near_rad', 0.07),
        goal_dist_std=vn.get('goal_dist_std_m', 0.08),
        goal_angle_std=vn.get('goal_angle_std_rad', 0.012),
        near_start=vn.get('near_start', 0.3),
        near_full=vn.get('near_full', 0.12),
        ball_dropout_far=vn.get('ball_dropout_far', 0.06),
        ball_dropout_near=vn.get('ball_dropout_near', 0.22),
        goal_dropout_far=vn.get('goal_dropout_far', 0.01),
        goal_dropout_near=vn.get('goal_dropout_near', 0.03),
        # Simulation
        max_steps=sim.get('max_steps', 500),
        num_substeps=num_substeps,
        # NN
        hidden_size=hidden_size,
        input_size=input_size,
        output_size=output_size,
        is_elman=is_elman,
        use_sigmoid=activation == 'sigmoid',
        use_floreano_mapping=wheel_mapping == 'floreano',
        sensor_normalization=sensor_normalization,
        use_x64=use_x64,
        validate_in_jax=validate_in_jax,
    )
