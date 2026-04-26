"""End-to-end test: run a short evolution and verify it works."""

import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_short_evolution():
    """Run 5 generations with pop=8, 2 trials, 50 steps — should complete fast."""
    from jax_sim.runner import run

    # Create a minimal test config
    config = {
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
            'global_seed': 42, 'max_steps': 50,
            'timestep': 0.1, 'physics_timestep': 0.01,
            'num_trials': 2, 'fitness_function': 'penalty_sparse',
        },
        'neural_network': {
            'type': 'elman', 'hidden_size': 5,
            'output_size': 2, 'vision_inputs': 4,
            'activation': 'sigmoid', 'wheel_output_mapping': 'floreano',
        },
        'curriculum': {'enabled': False},
        'challenge': {
            'enabled': True, 'parameter': 'forward_l4',
            'schedule': 'linear', 'total_generations': 5,
            'p_min': 0.1, 'mode': 'cumulative', 'k': 1,
            'area_correction': False,
        },
        'motor': {'noise': {'enabled': False}},
        'fitness_params': {
            'goal_bonus': 8.0, 'touch_bonus': 1.0,
            'approach_weight': 1.5, 'ball_to_goal_weight': 3.0,
            'goal_reward': 1.0, 'time_bonus': False,
        },
        'evolution': {
            'algorithm': 'ga', 'population_size': 8,
            'num_generations': 5, 'crossover_rate': 0.2,
            'crossover_type': 'single_point',
            'weight_bounds': [-2.0, 2.0],
            'initial_weight_range': [-0.5, 0.5],
            'elitism': 2, 'tournament_size': 2,
            'mutation_rate': 0.15, 'mutation_scale': 0.2,
            'mutation_type': 'gaussian', 'immigrant_frac': 0.0,
            'l2_lambda': 0.0, 'num_workers': 1,
        },
        'seeding': {'enabled': False},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config['results_dir'] = tmpdir
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run evolution
        run(config_path, batch_mode=True)

        # Check outputs exist
        assert os.path.exists(os.path.join(tmpdir, 'best_weights.npy'))
        assert os.path.exists(os.path.join(tmpdir, 'generation_stats.json'))
        assert os.path.exists(os.path.join(tmpdir, 'validation_summary.txt'))

        # Check generation stats
        import json
        with open(os.path.join(tmpdir, 'generation_stats.json')) as f:
            stats = json.load(f)
        assert len(stats) == 5, f"Expected 5 generations, got {len(stats)}"

        print(f"\nE2E test passed! 5 generations completed.")
        for i, s in enumerate(stats):
            print(f"  Gen {i+1}: best={s['best']:.2f} mean={s['mean']:.2f}")


if __name__ == '__main__':
    test_short_evolution()
