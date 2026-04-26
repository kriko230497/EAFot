"""Diagnostik: verificér at JAX-validering er konsistent med rollout.

Loader best_weights.npy, kører de præcis samme 50 scenarier som
JAX-valideringen brugte, og tjekker at goal-count matcher.

For hvert mål-trial: print final position, goal_scored_step, done_step.

Usage:
    conda activate sweep
    python3 jax_sim/tests/test_visualization_consistency.py [results_dir]

    Eksempel:
    python3 jax_sim/tests/test_visualization_consistency.py results/jax_hybrid2_rnn_iw4_jval_500g/seed_0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import tempfile

from jax_sim.config import load_config as load_config_jax
from jax_sim.simulator import rollout_with_trajectory
from jax_sim.challenge import sample_uniform_states
from jax_sim.neural_network import num_weights


def main():
    # Find results dir
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to latest jax_jval result
        results_dir = 'results/jax_hybrid2_rnn_iw4_jval_500g/seed_0'

    print(f"Results dir: {results_dir}")

    # Load weights
    weights_path = os.path.join(results_dir, 'best_weights.npy')
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found")
        return
    weights = np.load(weights_path)
    print(f"Weights: {weights_path} ({weights.shape})")

    # Load config
    cfg_files = [f for f in os.listdir(results_dir) if f.endswith('.yaml')]
    if not cfg_files:
        print("ERROR: No config yaml found in results dir")
        return
    cfg_path = os.path.join(results_dir, cfg_files[0])
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    print(f"Config: {cfg_path}")

    # Load validation summary for comparison
    val_path = os.path.join(results_dir, 'validation_summary.txt')
    reported_goals = None
    if os.path.exists(val_path):
        with open(val_path) as f:
            txt = f.read()
        import re
        m = re.search(r'Goals scored:\s+(\d+)/(\d+)', txt)
        if m:
            reported_goals = int(m.group(1))
            print(f"Reported validation: {m.group(1)}/{m.group(2)}")

    # Load trial details
    td_path = os.path.join(results_dir, 'trial_details.json')
    selected_label = None
    if os.path.exists(td_path):
        with open(td_path) as f:
            td = json.load(f)
        selected_label = td.get('selected_label', None)
        val_mode = td.get('validation', {}).get('mode', 'numpy')
        print(f"Selected candidate: {selected_label}")
        print(f"Validation mode: {val_mode}")

    # Load JAX params
    # Ensure jax section exists
    if 'jax' not in config:
        config['jax'] = {'use_x64': False, 'validate_in_jax': True}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        params = load_config_jax(tmp)
    finally:
        os.unlink(tmp)

    # Generate the same 50 uniform states as JAX validation used
    val_rng = np.random.RandomState(42)
    states = sample_uniform_states(50, config, val_rng)
    seed_offset = 100000

    dtype = jnp.float64 if params.use_x64 else jnp.float32
    weights_jax = jnp.array(weights, dtype=dtype)

    print(f"\nRunning 50 rollouts with trajectory...")
    print(f"{'Trial':>5s}  {'Goal':>4s}  {'GoalStep':>8s}  {'DoneStep':>8s}  "
          f"{'BallX':>7s}  {'BallY':>7s}  {'RobotX':>7s}  {'RobotY':>7s}")
    print("-" * 70)

    goals_counted = 0
    goal_trials = []

    for i in range(50):
        rx, ry, rq, bx, by = states[i]
        key = jax.random.PRNGKey(seed_offset + i)

        final, traj = rollout_with_trajectory(
            weights_jax, rx, ry, rq, bx, by, key, params)

        goal = int(final.goal_scored)
        goal_step = int(final.goal_scored_step)
        scored = goal == 1

        if scored:
            goals_counted += 1
            goal_trials.append(i)

        # Find effective done step
        done_flags = np.array(traj.done)
        if done_flags.any():
            done_step = int(np.argmax(done_flags))
        else:
            done_step = params.max_steps

        final_bx = float(final.ball.x)
        final_by = float(final.ball.y)
        final_rx = float(final.robot.x)
        final_ry = float(final.robot.y)

        # Print all goal trials + first 10
        if scored or i < 10:
            marker = "GOAL" if scored else "miss"
            print(f"{i:>5d}  {marker:>4s}  {goal_step:>8d}  {done_step:>8d}  "
                  f"{final_bx:>7.3f}  {final_by:>7.3f}  "
                  f"{final_rx:>7.3f}  {final_ry:>7.3f}")

    print("-" * 70)
    print(f"\nGoals from rollout: {goals_counted}/50")
    print(f"Goal trials: {goal_trials}")

    if reported_goals is not None:
        match = goals_counted == reported_goals
        print(f"Reported goals:     {reported_goals}/50")
        print(f"Match: {'YES' if match else 'NO — BUG!'}")

        if not match:
            print(f"\n  MISMATCH: rollout says {goals_counted} but validation said {reported_goals}")
            print(f"  This means the validation counted differently than a fresh rollout.")
            print(f"  Possible causes:")
            print(f"    - Non-determinism (different RNG path)")
            print(f"    - Different weights (validator selected a different candidate)")
            print(f"    - Bug in goal detection")
    else:
        print("(No reported goal count to compare against)")

    # Sanity check: for goal trials, verify ball actually crossed goal line
    print(f"\nGoal position sanity check:")
    hl = params.half_length
    hgw = params.half_goal_width
    for trial_idx in goal_trials:
        rx, ry, rq, bx, by = states[trial_idx]
        key = jax.random.PRNGKey(seed_offset + trial_idx)
        final, traj = rollout_with_trajectory(
            weights_jax, rx, ry, rq, bx, by, key, params)

        done_flags = np.array(traj.done)
        done_step = int(np.argmax(done_flags)) if done_flags.any() else params.max_steps

        # Ball position at done_step
        if done_step < params.max_steps:
            bx_at_done = float(traj.ball.x[done_step])
            by_at_done = float(traj.ball.y[done_step])
        else:
            bx_at_done = float(final.ball.x)
            by_at_done = float(final.ball.y)

        past_line = bx_at_done - params.ball_radius >= hl
        in_goal_width = abs(by_at_done) <= hgw
        valid_goal = past_line and in_goal_width

        print(f"  Trial {trial_idx:>3d}: ball=({bx_at_done:.4f}, {by_at_done:.4f})  "
              f"past_line={past_line}  in_width={in_goal_width}  "
              f"{'VALID' if valid_goal else 'SUSPICIOUS'}")


if __name__ == '__main__':
    main()
