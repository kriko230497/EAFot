"""Parity test: compare JAX vs NumPy simulation step by step.

Runs both implementations with identical initial conditions and NO noise,
then compares robot/ball positions at each step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

# NumPy implementation
from core.simulator import Simulator
from core.neural_network import NeuralNetworkController
from utils.config_loader import load_config as load_config_numpy

# JAX implementation
from jax_sim.config import load_config as load_config_jax
from jax_sim.simulator import rollout
from jax_sim.neural_network import num_weights


def make_test_config():
    """Create a minimal config dict with NO noise for parity testing."""
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
            'placement': 'fixed',
            'x_range': [-1.3665, 1.3665],
            'y_range': [-1.1865, 1.1865],
        },
        'teams': {
            'blue': {
                'num_robots': 1, 'placement': 'fixed',
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
            'global_seed': 0, 'max_steps': 50,
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
        'fitness_params': {
            'goal_bonus': 8.0, 'touch_bonus': 1.0,
            'approach_weight': 1.5, 'ball_to_goal_weight': 3.0,
            'goal_reward': 1.0, 'time_bonus': False,
        },
    }


def run_numpy(config, weights, rx, ry, rq, bx, by, max_steps):
    """Run NumPy simulation, return per-step positions."""
    config = dict(config)
    config['challenge'] = {'enabled': False}
    config['curriculum'] = {'enabled': False}
    config['simulation']['max_steps'] = max_steps

    sim = Simulator(config)

    # Reset spawns the robot and ball
    sim.reset(random_seed=0)

    # Override positions after reset
    sim.robots[0].x = rx
    sim.robots[0].y = ry
    sim.robots[0].q = rq
    sim.robots[0]._left_wheel_actual = 0.0
    sim.robots[0]._right_wheel_actual = 0.0
    sim.ball.x = bx
    sim.ball.y = by
    sim.ball.vx = 0.0
    sim.ball.vy = 0.0
    sim.ball.omega = 0.0

    # Set NN weights (NeuralNetworkController needs full config)
    nn = NeuralNetworkController(config)
    nn.set_weights(weights)
    sim.controllers[0] = nn
    nn.reset_state()

    # Record initial ball-to-goal distance
    target_goal_x = config['pitch']['length'] / 2.0
    init_btg = np.sqrt((bx - target_goal_x)**2 + by**2)
    sim.robot_states[0].initial_ball_to_goal = init_btg
    sim.robot_states[0].min_ball_to_goal = init_btg

    robot_positions = []
    ball_positions = []

    for step in range(max_steps):
        robot_positions.append((
            float(sim.robots[0].x),
            float(sim.robots[0].y),
            float(sim.robots[0].q),
        ))
        ball_positions.append((
            float(sim.ball.x),
            float(sim.ball.y),
            float(sim.ball.vx),
            float(sim.ball.vy),
        ))

        running = sim.step()
        if not running:
            break

    goal = sim.goal_scored

    return robot_positions, ball_positions, goal


def run_jax(config, weights, rx, ry, rq, bx, by, max_steps):
    """Run JAX simulation, return per-step positions."""
    import yaml, tempfile

    # Write config to temp file for load_config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name

    try:
        params = load_config_jax(tmp_path)
        params = params._replace(max_steps=max_steps)
    finally:
        os.unlink(tmp_path)

    weights_jax = jnp.array(weights, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)

    # We need step-by-step output, so run manually
    from jax_sim.simulator import init_state, step_fn

    state = init_state(rx, ry, rq, bx, by, key, params)

    robot_positions = []
    ball_positions = []

    for step in range(max_steps):
        robot_positions.append((
            float(state.robot.x),
            float(state.robot.y),
            float(state.robot.q),
        ))
        ball_positions.append((
            float(state.ball.x),
            float(state.ball.y),
            float(state.ball.vx),
            float(state.ball.vy),
        ))

        if bool(state.done):
            break

        state = step_fn(state, weights_jax, params)

    goal = int(state.goal_scored)

    return robot_positions, ball_positions, goal


def test_parity():
    """Compare NumPy and JAX simulations step by step."""
    config = make_test_config()
    max_steps = 50

    # Fixed initial positions
    rx, ry, rq = -0.3, 0.1, 0.5
    bx, by = 0.8, -0.1

    # Random but identical weights
    np.random.seed(42)
    n = num_weights(12, 5, 2)
    weights = np.random.randn(n).astype(np.float64) * 0.3

    print(f"Running parity test: {max_steps} steps, no noise")
    print(f"  Robot start: ({rx}, {ry}, {rq})")
    print(f"  Ball start:  ({bx}, {by})")
    print(f"  Weights: {n} params, seed=42")
    print()

    # Run both
    np_robot, np_ball, np_goal = run_numpy(config, weights, rx, ry, rq, bx, by, max_steps)
    jax_robot, jax_ball, jax_goal = run_jax(config, weights.astype(np.float32),
                                             rx, ry, rq, bx, by, max_steps)

    # Compare step by step
    min_steps = min(len(np_robot), len(jax_robot))
    print(f"NumPy ran {len(np_robot)} steps, JAX ran {len(jax_robot)} steps")
    print(f"NumPy goal: {np_goal}, JAX goal: {jax_goal}")
    print()

    print(f"{'Step':>4s}  {'NP robot x':>10s} {'JAX robot x':>11s} {'dx':>10s}  "
          f"{'NP ball x':>10s} {'JAX ball x':>11s} {'dx':>10s}  "
          f"{'NP ball vx':>10s} {'JAX ball vx':>11s} {'dvx':>10s}")
    print("-" * 120)

    max_robot_err = 0.0
    max_ball_pos_err = 0.0
    max_ball_vel_err = 0.0
    first_diverge_step = None

    for i in range(min_steps):
        np_rx, np_ry, np_rq = np_robot[i]
        jax_rx, jax_ry, jax_rq = jax_robot[i]
        np_bx, np_by, np_bvx, np_bvy = np_ball[i]
        jax_bx, jax_by, jax_bvx, jax_bvy = jax_ball[i]

        robot_err = max(abs(np_rx - jax_rx), abs(np_ry - jax_ry))
        ball_pos_err = max(abs(np_bx - jax_bx), abs(np_by - jax_by))
        ball_vel_err = max(abs(np_bvx - jax_bvx), abs(np_bvy - jax_bvy))

        max_robot_err = max(max_robot_err, robot_err)
        max_ball_pos_err = max(max_ball_pos_err, ball_pos_err)
        max_ball_vel_err = max(max_ball_vel_err, ball_vel_err)

        if first_diverge_step is None and (robot_err > 1e-4 or ball_pos_err > 1e-4):
            first_diverge_step = i

        # Print every 5th step + first 5 + any with large error
        if i < 5 or i % 5 == 0 or robot_err > 1e-3 or ball_pos_err > 1e-3:
            print(f"{i:4d}  {np_rx:10.6f} {jax_rx:11.6f} {np_rx-jax_rx:10.2e}  "
                  f"{np_bx:10.6f} {jax_bx:11.6f} {np_bx-jax_bx:10.2e}  "
                  f"{np_bvx:10.6f} {jax_bvx:11.6f} {np_bvx-jax_bvx:10.2e}")

    print()
    print(f"Max robot position error:  {max_robot_err:.2e}")
    print(f"Max ball position error:   {max_ball_pos_err:.2e}")
    print(f"Max ball velocity error:   {max_ball_vel_err:.2e}")
    if first_diverge_step is not None:
        print(f"First divergence (>1e-4):  step {first_diverge_step}")
    else:
        print(f"No divergence > 1e-4 detected!")

    print()
    if max_robot_err < 0.01 and max_ball_pos_err < 0.01:
        print("PASS — simulations are closely matched")
    elif max_robot_err < 0.1 and max_ball_pos_err < 0.1:
        print("MARGINAL — small differences, likely float32 vs float64")
    else:
        print("FAIL — significant divergence, physics likely differs")


if __name__ == '__main__':
    test_parity()
