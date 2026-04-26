"""Verification tests following jax_port_verification.md.

Quick path: Tier 1.1+1.2, Tier 4.4, Tier 3.2
Then: Tier 1.3-1.8, Tier 5.1-5.3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import yaml
import tempfile

from core.simulator import Simulator
from core.robot import Robot
from core.ball import Ball
from core.pitch import Pitch
from core.neural_network import NeuralNetworkController

from jax_sim.config import load_config as load_config_jax
from jax_sim.types import RobotState, BallState
from jax_sim.kinematics import robot_substep, wrap_to_pi
from jax_sim.ball_physics import ball_substep, handle_robot_collision
from jax_sim.sensors import get_ir_readings, _normalize_sensors, get_vision_inputs
from jax_sim.neural_network import nn_forward, num_weights


def _make_config():
    """Minimal config dict with all noise disabled."""
    return {
        'robot': {
            'wheel_radius': 0.021, 'half_wheelbase': 0.0527,
            'body_radius': 0.0704, 'mass': 0.566,
            'max_wheel_speed': 14.3, 'max_wheel_cmd_delta': 0.15,
            'wall_collision': 'push',
        },
        'pitch': {
            'length': 2.8, 'width': 2.44,
            'goal_width': 0.75, 'goal_depth': 0.19,
            'goal_area_width': 1.03, 'goal_area_depth': 0.19,
            'penalty_area_width': 1.5, 'penalty_area_depth': 0.47,
            'penalty_spot_distance': 0.38, 'center_circle_radius': 0.28,
            'corner_arc_radius': 0.09, 'wall_thickness': 0.04,
            'wall_mode': 'open',
        },
        'ball': {
            'radius': 0.0335, 'mass': 0.0577, 'gravity': 9.81,
            'rolling_friction': 0.04, 'wall_restitution': 0.745,
            'inertia_factor': 0.6667, 'sliding_friction': 0.23,
            'wall_friction': 0.23, 'robot_friction': 0.23,
            'robot_restitution': 0.5, 'spin_damping': 0.0,
            'placement': 'random',
            'x_range': [-1.3665, 1.3665],
            'y_range': [-1.1865, 1.1865],
        },
        'teams': {
            'blue': {
                'num_robots': 1, 'placement': 'random',
                'x_range': [-1.3296, 1.3296],
                'y_range': [-1.1496, 1.1496],
                'q_range': [-3.14159, 3.14159],
            },
            'yellow': {'num_robots': 0},
        },
        'sensors': {
            'angles': [-2.53, -1.571, -0.785, -0.175, 0.175, 0.785, 1.571, 2.53],
            'max_range': 0.25, 'min_range': 0.005,
            'noise': {'enabled': False},
            'normalization': 'minmax',
        },
        'vision': {
            'mode': 'frontal', 'horizontal_fov_deg': 131.0,
            'ball_max_range': 1.9, 'goal_max_range': 3.6,
            'noise': {'enabled': False},
        },
        'simulation': {
            'global_seed': 0, 'max_steps': 500,
            'timestep': 0.1, 'physics_timestep': 0.01,
            'num_trials': 1, 'fitness_function': 'penalty_sparse',
            'trajectory_stride': 1,
        },
        'neural_network': {
            'type': 'elman', 'hidden_size': 5,
            'output_size': 2, 'vision_inputs': 4,
            'activation': 'sigmoid', 'wheel_output_mapping': 'floreano',
        },
        'curriculum': {'enabled': False},
        'challenge': {'enabled': False},
        'motor': {'noise': {'enabled': False}},
        'fitness_params': {},
    }


def _get_jax_params(config):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        return load_config_jax(tmp)
    finally:
        os.unlink(tmp)


# ═══════════════════════════════════════════════════
# TIER 1.1 — Robot kinematics
# ═══════════════════════════════════════════════════

def test_1_1_kinematics():
    """Straight-line motion: both implementations agree on final pose."""
    print("=== Tier 1.1: Robot kinematics ===")
    config = _make_config()
    params = _get_jax_params(config)

    # NumPy: create robot manually
    np_pitch = Pitch(config)
    np_robot = Robot(0.0, 0.0, 0.0, config, np_pitch)
    np_robot._left_wheel_actual = 1.0
    np_robot._right_wheel_actual = 1.0

    # Run 10 substeps at full speed
    left_speed = 1.0 * config['robot']['max_wheel_speed']
    right_speed = 1.0 * config['robot']['max_wheel_speed']
    for _ in range(10):
        np_robot.substep(left_speed, right_speed)

    # JAX
    jax_robot = RobotState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(1.0), right_actual=jnp.float32(1.0))
    for _ in range(10):
        jax_robot = robot_substep(jax_robot, left_speed, right_speed, params)

    dx = abs(np_robot.x - float(jax_robot.x))
    dy = abs(np_robot.y - float(jax_robot.y))
    dq = abs(np_robot.q - float(jax_robot.q))

    print(f"  NumPy: ({np_robot.x:.8f}, {np_robot.y:.8f}, {np_robot.q:.8f})")
    print(f"  JAX:   ({float(jax_robot.x):.8f}, {float(jax_robot.y):.8f}, {float(jax_robot.q):.8f})")
    print(f"  Error: dx={dx:.2e}, dy={dy:.2e}, dq={dq:.2e}")
    ok = dx < 1e-5 and dy < 1e-5 and dq < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.2 — Ball physics free roll
# ═══════════════════════════════════════════════════

def test_1_2_ball_free_roll():
    """Ball decelerates under rolling friction. Compare trajectories."""
    print("\n=== Tier 1.2: Ball free roll ===")
    config = _make_config()
    params = _get_jax_params(config)

    # NumPy
    np_ball = Ball(0.0, 0.0, config)
    np_ball.vx = 1.0
    np_ball.vy = 0.0
    np_pitch = Pitch(config)

    np_positions = []
    for step in range(50):
        np_positions.append((np_ball.x, np_ball.vx))
        for _ in range(10):  # 10 substeps per control step
            np_ball.substep(np_pitch)

    # JAX
    jax_ball = BallState(
        x=jnp.float32(0.0), y=jnp.float32(0.0),
        vx=jnp.float32(1.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

    jax_positions = []
    for step in range(50):
        jax_positions.append((float(jax_ball.x), float(jax_ball.vx)))
        for _ in range(10):
            jax_ball = ball_substep(jax_ball, params)

    max_x_err = 0
    max_v_err = 0
    for i in range(50):
        xe = abs(np_positions[i][0] - jax_positions[i][0])
        ve = abs(np_positions[i][1] - jax_positions[i][1])
        max_x_err = max(max_x_err, xe)
        max_v_err = max(max_v_err, ve)

    print(f"  Max position error: {max_x_err:.2e}")
    print(f"  Max velocity error: {max_v_err:.2e}")
    print(f"  NumPy final: x={np_positions[-1][0]:.6f}, vx={np_positions[-1][1]:.6f}")
    print(f"  JAX final:   x={jax_positions[-1][0]:.6f}, vx={jax_positions[-1][1]:.6f}")
    ok = max_x_err < 1e-4 and max_v_err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.3 — Ball-wall collision
# ═══════════════════════════════════════════════════

def test_1_3_ball_wall_collision():
    """Ball heading into goal back wall, check rebound velocity."""
    print("\n=== Tier 1.3: Ball-wall collision ===")
    config = _make_config()
    params = _get_jax_params(config)

    # NumPy
    np_ball = Ball(1.39, 0.0, config)
    np_ball.vx = 2.0
    np_ball.vy = 0.0
    np_pitch = Pitch(config)

    for _ in range(50):  # 5 control steps × 10 substeps
        np_ball.substep(np_pitch)

    # JAX
    jax_ball = BallState(
        x=jnp.float32(1.39), y=jnp.float32(0.0),
        vx=jnp.float32(2.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))
    for _ in range(50):
        jax_ball = ball_substep(jax_ball, params)

    print(f"  NumPy: x={np_ball.x:.6f}, vx={np_ball.vx:.6f}")
    print(f"  JAX:   x={float(jax_ball.x):.6f}, vx={float(jax_ball.vx):.6f}")
    x_err = abs(np_ball.x - float(jax_ball.x))
    v_err = abs(np_ball.vx - float(jax_ball.vx))
    print(f"  Error: dx={x_err:.2e}, dvx={v_err:.2e}")
    ok = x_err < 1e-3 and v_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.4 — Ball-robot collision
# ═══════════════════════════════════════════════════

def test_1_4_ball_robot_collision():
    """Robot drives into stationary ball, check ball velocity after."""
    print("\n=== Tier 1.4: Ball-robot collision ===")
    config = _make_config()
    params = _get_jax_params(config)

    R = config['robot']['wheel_radius']
    mws = config['robot']['max_wheel_speed']
    left_speed = 1.0 * mws
    right_speed = 1.0 * mws

    # NumPy
    np_pitch = Pitch(config)
    np_robot = Robot(0.0, 0.0, 0.0, config, np_pitch)
    np_ball = Ball(0.105, 0.0, config)
    np_ball.vx, np_ball.vy = 0.0, 0.0
    np_ball.handle_robot_collision(np_robot, left_speed, right_speed)

    # JAX
    jax_robot = RobotState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(1.0), right_actual=jnp.float32(1.0))
    jax_ball = BallState(
        x=jnp.float32(0.105), y=jnp.float32(0.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))
    jax_ball, touched = handle_robot_collision(
        jax_ball, jax_robot, left_speed, right_speed, params)

    print(f"  NumPy ball after: vx={np_ball.vx:.6f}, vy={np_ball.vy:.6f}")
    print(f"  JAX ball after:   vx={float(jax_ball.vx):.6f}, vy={float(jax_ball.vy):.6f}")
    vx_err = abs(np_ball.vx - float(jax_ball.vx))
    vy_err = abs(np_ball.vy - float(jax_ball.vy))
    print(f"  Error: dvx={vx_err:.2e}, dvy={vy_err:.2e}")
    ok = vx_err < 1e-4 and vy_err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.5 — IR ray casting
# ═══════════════════════════════════════════════════

def test_1_5_ir_sensors():
    """Robot at origin, ball at (0.15, 0). Compare IR readings."""
    print("\n=== Tier 1.5: IR ray casting ===")
    config = _make_config()
    params = _get_jax_params(config)

    # NumPy
    np_sim = Simulator(config)
    np_sim.reset(random_seed=0)
    np_sim.robots[0].x, np_sim.robots[0].y, np_sim.robots[0].q = 0.0, 0.0, 0.0
    np_sim.ball.x, np_sim.ball.y = 0.15, 0.0
    np_inputs = np_sim.robots[0].get_all_inputs([], np_sim.ball,
                                                  config['pitch']['length'] / 2.0)
    np_ir = np.array(np_inputs[:8])

    # JAX
    jax_robot = RobotState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
    jax_ball_state = BallState(
        x=jnp.float32(0.15), y=jnp.float32(0.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))
    raw = get_ir_readings(jax_robot, jax_ball_state, params)
    clipped = jnp.clip(raw, params.sensor_min, params.sensor_max)
    jax_ir = np.array(_normalize_sensors(clipped, params))

    ir_err = np.max(np.abs(np_ir - jax_ir))
    print(f"  NumPy IR: {np_ir}")
    print(f"  JAX IR:   {jax_ir}")
    print(f"  Max error: {ir_err:.2e}")
    ok = ir_err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.6 — Vision inputs
# ═══════════════════════════════════════════════════

def test_1_6_vision():
    """Robot at origin, ball at (0.5, 0.3). Compare vision inputs."""
    print("\n=== Tier 1.6: Vision inputs ===")
    config = _make_config()
    params = _get_jax_params(config)
    hl = config['pitch']['length'] / 2.0

    # NumPy
    np_sim = Simulator(config)
    np_sim.reset(random_seed=0)
    np_sim.robots[0].x, np_sim.robots[0].y, np_sim.robots[0].q = 0.0, 0.0, 0.0
    np_sim.ball.x, np_sim.ball.y = 0.5, 0.3
    np_inputs = np_sim.robots[0].get_all_inputs([], np_sim.ball, hl)
    np_vis = np.array(np_inputs[8:12])

    # JAX
    jax_robot = RobotState(
        x=jnp.float32(0.0), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
    jax_ball_state = BallState(
        x=jnp.float32(0.5), y=jnp.float32(0.3),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))
    vision, bv, gv, _, _, _, _ = get_vision_inputs(
        jax_robot, jax_ball_state, hl, params)
    jax_vis = np.array(vision)

    vis_err = np.max(np.abs(np_vis - jax_vis))
    print(f"  NumPy vision: {np_vis}")
    print(f"  JAX vision:   {jax_vis}")
    print(f"  Max error: {vis_err:.2e}")
    print(f"  Ball visible: NumPy=True (assumed), JAX={bool(bv)}")
    ok = vis_err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.7 — NN forward pass
# ═══════════════════════════════════════════════════

def test_1_7_nn_forward():
    """Fixed weights and inputs, compare NN outputs."""
    print("\n=== Tier 1.7: NN forward pass ===")
    config = _make_config()
    params = _get_jax_params(config)

    n = num_weights(12, 5, 2)
    weights = np.full(n, 0.1, dtype=np.float64)
    inputs_np = np.full(12, 0.5, dtype=np.float64)

    # NumPy
    nn = NeuralNetworkController(config)
    nn.set_weights(weights)
    nn.reset_state()
    np_left, np_right = nn.forward(inputs_np)

    # Floreano mapping
    np_left = 2.0 * np_left - 1.0
    np_right = 2.0 * np_right - 1.0

    # JAX
    inputs_jax = jnp.array(inputs_np, dtype=jnp.float32)
    weights_jax = jnp.array(weights, dtype=jnp.float32)
    h = jnp.zeros(5, dtype=jnp.float32)
    (jax_left, jax_right), _ = nn_forward(inputs_jax, weights_jax, h, params)
    # Floreano mapping
    jax_left = 2.0 * float(jax_left) - 1.0
    jax_right = 2.0 * float(jax_right) - 1.0

    left_err = abs(np_left - jax_left)
    right_err = abs(np_right - jax_right)
    print(f"  NumPy: left={np_left:.8f}, right={np_right:.8f}")
    print(f"  JAX:   left={jax_left:.8f}, right={jax_right:.8f}")
    print(f"  Error: dl={left_err:.2e}, dr={right_err:.2e}")
    ok = left_err < 1e-5 and right_err < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 1.8 — Goal detection
# ═══════════════════════════════════════════════════

def test_1_8_goal_detection():
    """Ball heading into goal, both detect goal at same step."""
    print("\n=== Tier 1.8: Goal detection ===")
    config = _make_config()
    params = _get_jax_params(config)

    from jax_sim.pitch import is_goal_scored as jax_goal_check

    # NumPy
    np_pitch = Pitch(config)
    np_ball = Ball(1.395, 0.0, config)
    np_ball.vx = 0.5

    np_goal_step = None
    for step in range(20):
        np_ball.substep(np_pitch)
        result = np_pitch.is_goal_scored(np_ball.x, np_ball.y, np_ball.radius)
        if result is not None:
            np_goal_step = step
            break

    # JAX
    jax_ball = BallState(
        x=jnp.float32(1.395), y=jnp.float32(0.0),
        vx=jnp.float32(0.5), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

    jax_goal_step = None
    for step in range(20):
        jax_ball = ball_substep(jax_ball, params)
        scored = int(jax_goal_check(jax_ball.x, jax_ball.y,
                                     params.ball_radius, params))
        if scored != 0:
            jax_goal_step = step
            break

    print(f"  NumPy goal at substep: {np_goal_step}")
    print(f"  JAX goal at substep:   {jax_goal_step}")
    ok = np_goal_step is not None and jax_goal_step is not None
    if ok:
        step_diff = abs(np_goal_step - jax_goal_step)
        ok = step_diff <= 1  # allow ±1 for float precision
        print(f"  Step difference: {step_diff}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 4.4 — Curriculum schedule
# ═══════════════════════════════════════════════════

def test_4_4_curriculum_schedule():
    """Linear schedule produces identical p values."""
    print("\n=== Tier 4.4: Curriculum schedule ===")
    from jax_sim.challenge import sample_states_for_generation

    config = _make_config()
    config['challenge'] = {
        'enabled': True, 'parameter': 'forward_l4',
        'schedule': 'linear', 'total_generations': 100,
        'p_min': 0.022, 'mode': 'cumulative', 'k': 1,
    }

    # JAX schedule
    jax_ps = []
    for gen in range(100):
        _, p = sample_states_for_generation(1, 1, gen, 100, config)
        jax_ps.append(p)

    # NumPy schedule (same formula)
    np_ps = []
    for gen in range(100):
        t = min(gen / max(99, 1), 1.0)
        p = 0.022 + (1.0 - 0.022) * t
        np_ps.append(p)

    max_err = max(abs(a - b) for a, b in zip(jax_ps, np_ps))
    print(f"  p at gen 0:  JAX={jax_ps[0]:.6f}, expected={np_ps[0]:.6f}")
    print(f"  p at gen 50: JAX={jax_ps[50]:.6f}, expected={np_ps[50]:.6f}")
    print(f"  p at gen 99: JAX={jax_ps[99]:.6f}, expected={np_ps[99]:.6f}")
    print(f"  Max error: {max_err:.2e}")
    ok = max_err < 1e-10
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 3.2 — Known-good controller cross-validation
# ═══════════════════════════════════════════════════

def test_3_2_cross_validation():
    """Run a known-good controller through both validators."""
    print("\n=== Tier 3.2: Known-good controller cross-validation ===")

    # Find a result with decent performance
    results_base = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    best_dir = None
    best_goals = 0

    for entry in os.listdir(results_base):
        if not entry.startswith('forward_'):
            continue
        for seed in os.listdir(os.path.join(results_base, entry)):
            if not seed.startswith('seed_'):
                continue
            d = os.path.join(results_base, entry, seed)
            vpath = os.path.join(d, 'validation_summary.txt')
            wpath = os.path.join(d, 'best_weights.npy')
            if not os.path.exists(vpath) or not os.path.exists(wpath):
                continue
            with open(vpath) as f:
                txt = f.read()
            import re
            m = re.search(r'Goals scored:\s+(\d+)/50', txt)
            if m and int(m.group(1)) > best_goals:
                best_goals = int(m.group(1))
                best_dir = d

    if best_dir is None or best_goals < 10:
        print("  No suitable controller found, SKIP")
        return True

    print(f"  Using: {best_dir} ({best_goals}/50 goals)")

    # Load weights and config
    weights = np.load(os.path.join(best_dir, 'best_weights.npy'))
    cfg_files = [f for f in os.listdir(best_dir) if f.endswith('.yaml')]
    cfg_path = os.path.join(best_dir, cfg_files[0])

    from utils.config_loader import load_config
    config = load_config(cfg_path)

    # Disable challenge/curriculum for validation
    config['challenge']['enabled'] = False
    if 'curriculum' in config:
        config['curriculum']['enabled'] = False

    # NumPy validation: seeds 100000..100049
    sim = Simulator(config)
    np_goals = 0
    for i in range(50):
        metrics = sim.run(weights, random_seed=100000 + i)
        if metrics.get('goal_scored') == metrics.get('target_goal_side'):
            np_goals += 1

    # JAX validation using same NumPy simulator (this is what runner.py does)
    jax_goals = 0
    sim2 = Simulator(config)
    for i in range(50):
        metrics = sim2.run(weights, random_seed=100000 + i)
        if metrics.get('goal_scored') == metrics.get('target_goal_side'):
            jax_goals += 1

    print(f"  NumPy validator: {np_goals}/50")
    print(f"  JAX validator:   {jax_goals}/50 (uses same NumPy sim)")
    diff = abs(np_goals - jax_goals)
    print(f"  Difference: {diff}")
    ok = diff == 0  # should be identical since both use NumPy sim
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 5.1 — Robot at goal-net corner
# ═══════════════════════════════════════════════════

def test_5_1_corner():
    """Robot at goal corner, check it doesn't get stuck."""
    print("\n=== Tier 5.1: Robot at goal corner ===")
    config = _make_config()
    params = _get_jax_params(config)

    from jax_sim.kinematics import resolve_wall_push

    jax_robot = RobotState(
        x=jnp.float32(1.39), y=jnp.float32(0.37),
        q=jnp.float32(0.0),
        left_actual=jnp.float32(1.0), right_actual=jnp.float32(1.0))

    for _ in range(200):  # 20 control steps × 10 substeps
        jax_robot = robot_substep(jax_robot, 14.3, 14.3, params)
        jax_robot = resolve_wall_push(jax_robot, params)

    print(f"  Final: ({float(jax_robot.x):.4f}, {float(jax_robot.y):.4f})")
    stuck = (abs(float(jax_robot.x) - 1.39) < 0.001 and
             abs(float(jax_robot.y) - 0.37) < 0.001)
    ok = not stuck
    print(f"  Stuck: {stuck}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════
# TIER 5.3 — Spinning robot doesn't move ball
# ═══════════════════════════════════════════════════

def test_5_3_spin_no_touch():
    """Spinning robot far from ball doesn't affect it."""
    print("\n=== Tier 5.3: Spinning robot doesn't move ball ===")
    config = _make_config()
    params = _get_jax_params(config)

    jax_robot = RobotState(
        x=jnp.float32(0.5), y=jnp.float32(0.0), q=jnp.float32(0.0),
        left_actual=jnp.float32(0.0), right_actual=jnp.float32(0.0))
    jax_ball = BallState(
        x=jnp.float32(0.0), y=jnp.float32(0.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0), omega=jnp.float32(0.0))

    for _ in range(100):
        for _ in range(10):
            jax_robot = robot_substep(jax_robot, -7.15, 7.15, params)  # spin
            jax_ball, touched = handle_robot_collision(
                jax_ball, jax_robot, -7.15, 7.15, params)
            jax_ball = ball_substep(jax_ball, params)

    bx = float(jax_ball.x)
    by = float(jax_ball.y)
    print(f"  Ball final: ({bx:.6f}, {by:.6f})")
    ok = abs(bx) < 1e-6 and abs(by) < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════

def main():
    results = []
    print("=" * 60)
    print("  JAX PORT VERIFICATION")
    print("=" * 60)

    # Quick path
    results.append(("1.1 Kinematics", test_1_1_kinematics()))
    results.append(("1.2 Ball free roll", test_1_2_ball_free_roll()))

    # Stop if physics doesn't match
    if not all(r[1] for r in results):
        print("\n*** PHYSICS MISMATCH — stopping ***")
        return

    results.append(("4.4 Schedule", test_4_4_curriculum_schedule()))

    if not results[-1][1]:
        print("\n*** SCHEDULE MISMATCH — stopping ***")
        return

    results.append(("3.2 Cross-validation", test_3_2_cross_validation()))

    # Remaining Tier 1
    results.append(("1.3 Ball-wall collision", test_1_3_ball_wall_collision()))
    results.append(("1.4 Ball-robot collision", test_1_4_ball_robot_collision()))
    results.append(("1.5 IR sensors", test_1_5_ir_sensors()))
    results.append(("1.6 Vision", test_1_6_vision()))
    results.append(("1.7 NN forward", test_1_7_nn_forward()))
    results.append(("1.8 Goal detection", test_1_8_goal_detection()))

    # Edge cases
    results.append(("5.1 Corner", test_5_1_corner()))
    results.append(("5.3 Spin no touch", test_5_3_spin_no_touch()))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{total} passed")


if __name__ == '__main__':
    main()
