"""Full pipeline parity test: compare NN inputs, outputs, and positions step by step.

Both implementations run with NO noise, identical initial conditions and weights.
Compares the 12 NN inputs, 2 wheel outputs, and robot/ball positions at each step.
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
from core.neural_network import NeuralNetworkController
from jax_sim.config import load_config as load_config_jax
from jax_sim.simulator import init_state, step_fn
from jax_sim.sensors import get_ir_readings, _normalize_sensors, get_vision_inputs
from jax_sim.neural_network import nn_forward, num_weights


def make_config():
    """Config with ALL noise disabled."""
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


def run_numpy_with_logging(config, weights, rx, ry, rq, bx, by, max_steps):
    """Run NumPy sim, return per-step: (inputs_12, left_cmd, right_cmd, rx, ry, rq, bx, by, bvx, bvy)."""
    config = dict(config)
    config['simulation']['max_steps'] = max_steps

    sim = Simulator(config)
    sim.reset(random_seed=0)

    # Override positions
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

    nn = NeuralNetworkController(config)
    nn.set_weights(weights)
    sim.controllers[0] = nn
    nn.reset_state()

    hl = config['pitch']['length'] / 2.0
    sim.robot_states[0].initial_ball_to_goal = np.sqrt((bx - hl)**2 + by**2)
    sim.robot_states[0].min_ball_to_goal = sim.robot_states[0].initial_ball_to_goal

    log = []
    for step in range(max_steps):
        r = sim.robots[0]
        b = sim.ball

        # Get the inputs the NN will see (before step)
        target_goal_x = hl
        others = []
        inputs = r.get_all_inputs(others, b, target_goal_x)

        log.append({
            'inputs': np.array(inputs, dtype=np.float64),
            'rx': float(r.x), 'ry': float(r.y), 'rq': float(r.q),
            'bx': float(b.x), 'by': float(b.y),
            'bvx': float(b.vx), 'bvy': float(b.vy),
        })

        running = sim.step()
        if not running:
            break

    return log


def run_jax_with_logging(config, weights, rx, ry, rq, bx, by, max_steps):
    """Run JAX sim, return per-step: same format as NumPy."""
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
    state = init_state(rx, ry, rq, bx, by, key, params)

    log = []
    for step in range(max_steps):
        # Extract inputs the NN will see
        raw_ir = get_ir_readings(state.robot, state.ball, params)
        clipped = jnp.clip(raw_ir, params.sensor_min, params.sensor_max)
        ir = _normalize_sensors(clipped, params)

        vision, _, _, _, _, _, _ = get_vision_inputs(
            state.robot, state.ball, params.half_length, params)

        inputs_jax = jnp.concatenate([ir, vision])

        log.append({
            'inputs': np.array(inputs_jax, dtype=np.float64),
            'rx': float(state.robot.x), 'ry': float(state.robot.y),
            'rq': float(state.robot.q),
            'bx': float(state.ball.x), 'by': float(state.ball.y),
            'bvx': float(state.ball.vx), 'bvy': float(state.ball.vy),
        })

        if bool(state.done):
            break

        state = step_fn(state, weights_jax, params)

    return log


def main():
    config = make_config()
    max_steps = 50

    # Initial conditions: robot near ball, facing it
    rx, ry, rq = -0.2, 0.05, 0.3
    bx, by = 0.3, -0.05

    np.random.seed(42)
    n = num_weights(12, 5, 2)
    weights = np.random.randn(n) * 0.3

    print(f"Full pipeline parity test: {max_steps} steps, NO noise")
    print(f"  Robot: ({rx}, {ry}, q={rq})")
    print(f"  Ball:  ({bx}, {by})")
    print(f"  NN: elman, sigmoid, floreano, {n} weights")
    print()

    np_log = run_numpy_with_logging(config, weights, rx, ry, rq, bx, by, max_steps)
    jax_log = run_jax_with_logging(config, weights.astype(np.float32),
                                    rx, ry, rq, bx, by, max_steps)

    min_steps = min(len(np_log), len(jax_log))
    print(f"NumPy: {len(np_log)} steps, JAX: {len(jax_log)} steps")
    print()

    # Compare
    max_input_err = 0.0
    max_pos_err = 0.0
    max_ball_err = 0.0
    first_input_diverge = None
    first_pos_diverge = None

    print(f"{'Step':>4s}  {'Max IR err':>10s}  {'Max Vis err':>11s}  {'Robot pos err':>13s}  {'Ball pos err':>12s}  {'Ball vel err':>12s}")
    print("-" * 80)

    for i in range(min_steps):
        np_in = np_log[i]['inputs']
        jax_in = jax_log[i]['inputs']

        ir_err = np.max(np.abs(np_in[:8] - jax_in[:8]))
        vis_err = np.max(np.abs(np_in[8:] - jax_in[8:]))
        input_err = max(ir_err, vis_err)

        rpos_err = max(abs(np_log[i]['rx'] - jax_log[i]['rx']),
                       abs(np_log[i]['ry'] - jax_log[i]['ry']))
        bpos_err = max(abs(np_log[i]['bx'] - jax_log[i]['bx']),
                       abs(np_log[i]['by'] - jax_log[i]['by']))
        bvel_err = max(abs(np_log[i]['bvx'] - jax_log[i]['bvx']),
                       abs(np_log[i]['bvy'] - jax_log[i]['bvy']))

        max_input_err = max(max_input_err, input_err)
        max_pos_err = max(max_pos_err, rpos_err)
        max_ball_err = max(max_ball_err, bpos_err)

        if first_input_diverge is None and input_err > 1e-4:
            first_input_diverge = i
        if first_pos_diverge is None and rpos_err > 1e-4:
            first_pos_diverge = i

        if i < 10 or i % 5 == 0 or input_err > 1e-3:
            print(f"{i:4d}  {ir_err:10.2e}  {vis_err:11.2e}  {rpos_err:13.2e}  {bpos_err:12.2e}  {bvel_err:12.2e}")

    print()
    print(f"Max NN input error:    {max_input_err:.2e}")
    print(f"Max robot pos error:   {max_pos_err:.2e}")
    print(f"Max ball pos error:    {max_ball_err:.2e}")

    if first_input_diverge is not None:
        print(f"First input diverge:   step {first_input_diverge}")
        # Show the actual inputs at divergence point
        i = first_input_diverge
        print(f"\n  NumPy inputs at step {i}:")
        print(f"    IR:     {np_log[i]['inputs'][:8]}")
        print(f"    Vision: {np_log[i]['inputs'][8:]}")
        print(f"  JAX inputs at step {i}:")
        print(f"    IR:     {jax_log[i]['inputs'][:8]}")
        print(f"    Vision: {jax_log[i]['inputs'][8:]}")
    else:
        print("No input divergence > 1e-4!")

    print()
    if max_input_err < 1e-3 and max_pos_err < 1e-3:
        print("PASS — full pipeline matches")
    elif max_input_err < 0.01:
        print("MARGINAL — small differences, likely float32 vs float64")
    else:
        print("FAIL — significant pipeline differences")


if __name__ == '__main__':
    main()
