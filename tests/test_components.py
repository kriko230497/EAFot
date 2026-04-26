"""Unit tests for JAX simulation components."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from jax_sim.types import RobotState, BallState, StaticParams
from jax_sim.pitch import (build_goal_segments, is_goal_scored,
                            is_ball_out_of_play, ray_cast_multi)
from jax_sim.kinematics import robot_substep, wrap_to_pi, rate_limit_wheels
from jax_sim.ball_physics import ball_substep, handle_robot_collision
from jax_sim.neural_network import nn_forward, num_weights
from jax_sim.sensors import _ray_circle_distances_batch, _normalize_sensors


def _make_params():
    """Create a minimal StaticParams for testing."""
    seg_x1, seg_y1, seg_dx, seg_dy, seg_len_sq = build_goal_segments(
        1.4, 1.22, 0.375, 0.19)

    return StaticParams(
        wheel_radius=0.021, half_wheelbase=0.0527, body_radius=0.0704,
        robot_mass=0.566, max_wheel_speed=14.3, max_cmd_delta=0.15,
        ball_radius=0.0335, ball_mass=0.0577,
        ball_inertia=0.6667 * 0.0577 * 0.0335 ** 2,
        translational_decel=0.04 * 9.81,
        wall_restitution=0.745, robot_restitution=0.5,
        wall_friction=0.23, robot_friction=0.23, spin_damping=0.0,
        physics_timestep=0.01,
        half_length=1.4, half_width=1.22,
        half_goal_width=0.375, goal_depth=0.19,
        seg_x1=jnp.array(seg_x1), seg_y1=jnp.array(seg_y1),
        seg_dx=jnp.array(seg_dx), seg_dy=jnp.array(seg_dy),
        seg_len_sq=jnp.array(seg_len_sq), num_segments=len(seg_x1),
        sensor_angles=jnp.array([-2.53, -1.571, -0.785, -0.175,
                                  0.175, 0.785, 1.571, 2.53]),
        sensor_max=0.25, sensor_min=0.005,
        vision_half_fov=jnp.radians(131.0) / 2.0,
        ball_max_range=1.9, goal_max_range=3.6,
        sensor_noise_enabled=False, sensor_relative_std=0.04,
        sensor_bias_std=0.015, sensor_rho=0.92,
        motor_noise_enabled=False, motor_relative_std=0.03,
        motor_bias_std=0.02, motor_rho=0.95,
        motor_gain_clip_lo=0.75, motor_gain_clip_hi=1.25,
        vision_noise_enabled=False, vision_rho=0.95,
        ball_dist_bias_std=0.015, ball_angle_bias_std=0.01,
        goal_dist_bias_std=0.02, goal_angle_bias_std=0.006,
        ball_dist_std_far=0.1, ball_dist_std_near=0.16,
        ball_angle_std_far=0.015, ball_angle_std_near=0.07,
        goal_dist_std=0.08, goal_angle_std=0.012,
        near_start=0.3, near_full=0.12,
        ball_dropout_far=0.06, ball_dropout_near=0.22,
        goal_dropout_far=0.01, goal_dropout_near=0.03,
        max_steps=500, num_substeps=10,
        hidden_size=5, input_size=12, output_size=2,
        is_elman=False, use_sigmoid=False, use_floreano_mapping=False,
        sensor_normalization=0,
        use_x64=False,
        validate_in_jax=False,
    )


# ── Pitch Tests ──

class TestPitch:
    def test_goal_segments_count(self):
        x1, y1, dx, dy, lsq = build_goal_segments(1.4, 1.22, 0.375, 0.19)
        assert len(x1) == 6, "Should have 6 goal segments"

    def test_goal_scored_right(self):
        params = _make_params()
        # Ball fully past right goal line, within goal width
        result = is_goal_scored(1.5, 0.0, 0.0335, params)
        assert int(result) == 1

    def test_goal_scored_left(self):
        params = _make_params()
        result = is_goal_scored(-1.5, 0.0, 0.0335, params)
        assert int(result) == -1

    def test_no_goal_center(self):
        params = _make_params()
        result = is_goal_scored(0.0, 0.0, 0.0335, params)
        assert int(result) == 0

    def test_ball_out_sideline(self):
        params = _make_params()
        assert bool(is_ball_out_of_play(0.0, 1.3, 0.0335, params))

    def test_ball_not_out_inside(self):
        params = _make_params()
        assert not bool(is_ball_out_of_play(0.0, 0.0, 0.0335, params))

    def test_ball_out_endline_outside_goal(self):
        params = _make_params()
        # Past endline but outside goal width
        assert bool(is_ball_out_of_play(1.5, 0.8, 0.0335, params))


# ── Kinematics Tests ──

class TestKinematics:
    def test_straight_line(self):
        """Both wheels same speed = straight line."""
        params = _make_params()
        robot = RobotState(x=0.0, y=0.0, q=0.0,
                           left_actual=0.0, right_actual=0.0)
        robot = robot_substep(robot, 10.0, 10.0, params)
        assert float(robot.x) > 0.0, "Should move forward"
        assert abs(float(robot.y)) < 1e-6, "Should not move sideways"

    def test_turn_in_place(self):
        """Opposite wheel speeds = turn in place."""
        params = _make_params()
        robot = RobotState(x=0.0, y=0.0, q=0.0,
                           left_actual=0.0, right_actual=0.0)
        robot = robot_substep(robot, -5.0, 5.0, params)
        assert abs(float(robot.x)) < 1e-6
        assert abs(float(robot.y)) < 1e-6
        assert float(robot.q) != 0.0, "Should rotate"

    def test_wrap_to_pi(self):
        assert abs(float(wrap_to_pi(jnp.float32(4.0))) - (4.0 - 2 * np.pi)) < 1e-5
        assert abs(float(wrap_to_pi(jnp.float32(-4.0))) - (-4.0 + 2 * np.pi)) < 1e-5

    def test_rate_limiting(self):
        la, ra = rate_limit_wheels(1.0, 1.0, 0.0, 0.0, 0.15)
        assert abs(float(la) - 0.15) < 1e-6
        assert abs(float(ra) - 0.15) < 1e-6


# ── Ball Physics Tests ──

class TestBallPhysics:
    def test_ball_moves(self):
        params = _make_params()
        ball = BallState(x=0.0, y=0.0, vx=1.0, vy=0.0, omega=0.0)
        new_ball = ball_substep(ball, params)
        assert float(new_ball.x) > 0.0

    def test_ball_friction_slows(self):
        params = _make_params()
        ball = BallState(x=0.0, y=0.0, vx=0.5, vy=0.0, omega=0.0)
        new_ball = ball_substep(ball, params)
        speed_before = 0.5
        speed_after = jnp.sqrt(new_ball.vx ** 2 + new_ball.vy ** 2)
        assert float(speed_after) < speed_before

    def test_robot_ball_collision(self):
        params = _make_params()
        # Robot touching ball
        robot = RobotState(x=0.0, y=0.0, q=0.0,
                           left_actual=0.5, right_actual=0.5)
        ball = BallState(x=0.1, y=0.0, vx=0.0, vy=0.0, omega=0.0)
        new_ball, touched = handle_robot_collision(
            ball, robot, 7.0, 7.0, params)
        assert bool(touched), "Should detect collision"
        assert float(new_ball.vx) > 0.0, "Ball should be pushed away"

    def test_no_collision_when_far(self):
        params = _make_params()
        robot = RobotState(x=0.0, y=0.0, q=0.0,
                           left_actual=0.0, right_actual=0.0)
        ball = BallState(x=1.0, y=1.0, vx=0.0, vy=0.0, omega=0.0)
        new_ball, touched = handle_robot_collision(
            ball, robot, 0.0, 0.0, params)
        assert not bool(touched)


# ── Neural Network Tests ──

class TestNeuralNetwork:
    def test_feedforward_output_range(self):
        params = _make_params()  # use_sigmoid=False, tanh
        n = num_weights(12, 5, 2)
        weights = jnp.zeros(n)
        inputs = jnp.zeros(12)
        h = jnp.zeros(5)
        (left, right), h_new = nn_forward(inputs, weights, h, params)
        assert -1.0 <= float(left) <= 1.0
        assert -1.0 <= float(right) <= 1.0

    def test_weight_count(self):
        n = num_weights(12, 5, 2)
        # 12*5 + 5*5 + 5 + 5*2 + 2 = 60 + 25 + 5 + 10 + 2 = 102
        assert n == 102

    def test_elman_uses_hidden_state(self):
        params = _make_params()._replace(is_elman=True)
        n = num_weights(12, 5, 2)
        key = jax.random.PRNGKey(0)
        weights = jax.random.normal(key, (n,)) * 0.1
        inputs = jnp.ones(12) * 0.5
        h0 = jnp.zeros(5)

        (l1, r1), h1 = nn_forward(inputs, weights, h0, params)
        (l2, r2), h2 = nn_forward(inputs, weights, h1, params)

        # With recurrence and nonzero hidden state, outputs should differ
        assert not jnp.allclose(jnp.array([l1, r1]), jnp.array([l2, r2]))


# ── Sensor Tests ──

class TestSensors:
    def test_ray_circle_hit(self):
        # Ray from origin pointing right, circle at (1, 0) with radius 0.1
        directions = jnp.array([0.0])
        dists = _ray_circle_distances_batch(0.0, 0.0, directions, 1.0, 0.0, 0.1)
        assert float(dists[0]) < 1.1
        assert float(dists[0]) > 0.8

    def test_ray_circle_miss(self):
        # Ray pointing right, circle above
        directions = jnp.array([0.0])
        dists = _ray_circle_distances_batch(0.0, 0.0, directions, 1.0, 1.0, 0.1)
        assert float(dists[0]) == float('inf') or float(dists[0]) > 100

    def test_normalize_minmax(self):
        params = _make_params()
        dists = jnp.array([0.005, 0.125, 0.25])
        norm = _normalize_sensors(dists, params)
        assert abs(float(norm[0])) < 1e-5  # min -> 0
        assert abs(float(norm[2]) - 1.0) < 1e-5  # max -> 1


# ── Integration Tests ──

class TestIntegration:
    def test_single_rollout(self):
        """Run a single rollout without crashing."""
        from jax_sim.simulator import rollout
        params = _make_params()
        params = params._replace(max_steps=10)  # short for testing
        n = num_weights(12, 5, 2)
        weights = jnp.zeros(n)
        key = jax.random.PRNGKey(42)
        final = rollout(weights, 0.0, 0.0, 0.0, 0.5, 0.0, key, params)
        assert final.step == 10

    def test_rollout_with_random_weights(self):
        """Rollout with random weights shouldn't crash."""
        from jax_sim.simulator import rollout
        params = _make_params()
        params = params._replace(max_steps=20)
        n = num_weights(12, 5, 2)
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        weights = jax.random.normal(k1, (n,)) * 0.5
        final = rollout(weights, -0.5, 0.3, 1.0, 0.8, -0.2, k2, params)
        assert final.step == 20

    def test_vmap_over_trials(self):
        """vmap over multiple trials."""
        from jax_sim.simulator import rollout
        params = _make_params()
        params = params._replace(max_steps=5)
        n = num_weights(12, 5, 2)
        weights = jnp.zeros(n)

        def run_trial(key):
            return rollout(weights, 0.0, 0.0, 0.0, 0.5, 0.0, key, params)

        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        finals = jax.vmap(run_trial)(keys)
        assert finals.step.shape == (3,)


# ── Evolution Tests ──

class TestEvolution:
    def test_initialize_population(self):
        from jax_sim.evolution import initialize_population
        key = jax.random.PRNGKey(0)
        pop = initialize_population(key, 10, 50, -2.0, 2.0)
        assert pop.shape == (10, 50)
        assert float(jnp.min(pop)) >= -2.0
        assert float(jnp.max(pop)) <= 2.0

    def test_tournament_selection(self):
        from jax_sim.evolution import tournament_selection
        key = jax.random.PRNGKey(0)
        pop = jnp.zeros((10, 5))
        fitness = jnp.arange(10, dtype=jnp.float32)
        selected = tournament_selection(key, pop, fitness, 3)
        assert selected.shape == (10,)
        # Best fitness individual (idx 9) should be selected frequently
        # but this is stochastic so just check shape

    def test_gaussian_mutation(self):
        from jax_sim.evolution import gaussian_mutation
        key = jax.random.PRNGKey(0)
        ind = jnp.zeros(50)
        mutated = gaussian_mutation(key, ind, 0.15, 0.2, -2.0, 2.0)
        # Some genes should have changed
        assert not jnp.allclose(ind, mutated)
        # All within bounds
        assert float(jnp.min(mutated)) >= -2.0
        assert float(jnp.max(mutated)) <= 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
