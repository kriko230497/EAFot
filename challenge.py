"""Challenge parameter sampling — pre-generate initial states in NumPy.

The forward_l4 sampling uses rejection (for boundary validation) which is
inherently sequential. We pre-sample all states in NumPy, then pass them
as arrays into the JAX evaluation.
"""

import numpy as np


def sample_forward_l4_states(num_states, p, config, rng=None):
    """Sample initial states using forward_l4 challenge method.

    Args:
        num_states: how many valid states to generate
        p: difficulty parameter in [0, 1]
        config: full YAML config dict
        rng: numpy RandomState (optional)

    Returns:
        (num_states, 5) array of (rx, ry, rq, bx, by)
    """
    if rng is None:
        rng = np.random.RandomState()

    challenge = config.get('challenge', {})
    k = float(challenge.get('k', 4))
    area_correction = challenge.get('area_correction', False)

    pitch = config['pitch']
    ball = config['ball']
    robot = config['robot']
    teams = config.get('teams', {})
    blue = teams.get('blue', {})

    hl = pitch['length'] / 2.0
    hw = pitch['width'] / 2.0
    ball_radius = ball.get('radius', 0.0335)
    robot_radius = robot.get('body_radius', 0.0704)

    # Ranges from config
    bx_min, bx_max = ball.get('x_range', [-hl, hl])
    by_min, by_max = ball.get('y_range', [-hw, hw])
    rx_min, rx_max = blue.get('x_range', [-hl, hl])
    ry_min, ry_max = blue.get('y_range', [-hw, hw])
    rq_min, rq_max = blue.get('q_range', [-np.pi, np.pi])

    # Goal reference (right goal)
    goal_x = hl + ball_radius
    goal_y = 0.0

    # Max distances
    bg_dx_max = max(abs(bx_max - goal_x), abs(bx_min - goal_x))
    bg_dy_max = max(abs(by_max - goal_y), abs(by_min - goal_y))
    d_bg_max = np.sqrt(bg_dx_max ** 2 + bg_dy_max ** 2)

    rb_dx_max = max(abs(rx_max - bx_min), abs(rx_min - bx_max))
    rb_dy_max = max(abs(ry_max - by_min), abs(ry_min - by_max))
    d_rb_max = np.sqrt(rb_dx_max ** 2 + rb_dy_max ** 2)

    d_min = robot_radius + ball_radius + 0.001

    states = []
    max_attempts = num_states * 200

    for _ in range(max_attempts):
        if len(states) >= num_states:
            break

        # Step 1: r_bg
        r_bg_max = min(1.0, (2.0 * p ** k) ** (1.0 / k))
        if area_correction:
            r_bg = r_bg_max * np.sqrt(rng.uniform())
        else:
            r_bg = rng.uniform(0.0, r_bg_max)

        # Step 2: r_rb budget
        r_rb_budget = (2.0 * p ** k - r_bg ** k) ** (1.0 / k)
        r_rb_max = min(1.0, r_rb_budget)
        if area_correction:
            r_rb = r_rb_max * np.sqrt(rng.uniform())
        else:
            r_rb = rng.uniform(0.0, r_rb_max)

        # Step 3: ball position (polar from goal)
        d_bg = r_bg * d_bg_max
        theta_ball = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        bx = goal_x - np.cos(theta_ball) * d_bg
        by = goal_y + np.sin(theta_ball) * d_bg

        # Step 4: robot position (polar from ball)
        d_rb = d_min + r_rb * (d_rb_max - d_min)
        theta_robot = rng.uniform(0.0, 2.0 * np.pi)
        rx = bx + np.cos(theta_robot) * d_rb
        ry = by + np.sin(theta_robot) * d_rb
        rq = rng.uniform(rq_min, rq_max)

        # Step 5: boundary validation
        if not (bx_min <= bx <= bx_max and by_min <= by <= by_max):
            continue
        if not (rx_min <= rx <= rx_max and ry_min <= ry <= ry_max):
            continue

        states.append((rx, ry, rq, bx, by))

    if len(states) < num_states:
        raise RuntimeError(
            f"Only sampled {len(states)}/{num_states} states after {max_attempts} attempts")

    return np.array(states[:num_states], dtype=np.float32)


def sample_uniform_states(num_states, config, rng=None):
    """Sample uniform states inside config ranges (for validation).

    Matches the NumPy version's distribution when challenge.enabled = False.
    Robot and ball placed uniformly in their ranges with min-distance constraint.
    """
    if rng is None:
        rng = np.random.RandomState()

    pitch = config['pitch']
    ball = config['ball']
    robot = config['robot']
    blue = config.get('teams', {}).get('blue', {})

    hl = pitch['length'] / 2.0
    hw = pitch['width'] / 2.0
    ball_radius = ball.get('radius', 0.0335)
    robot_radius = robot.get('body_radius', 0.0704)

    bx_min, bx_max = ball.get('x_range', [-hl, hl])
    by_min, by_max = ball.get('y_range', [-hw, hw])
    rx_min, rx_max = blue.get('x_range', [-hl, hl])
    ry_min, ry_max = blue.get('y_range', [-hw, hw])
    rq_min, rq_max = blue.get('q_range', [-np.pi, np.pi])

    d_min = robot_radius + ball_radius + 0.001

    states = []
    max_attempts = num_states * 100
    for _ in range(max_attempts):
        if len(states) >= num_states:
            break
        bx = rng.uniform(bx_min, bx_max)
        by = rng.uniform(by_min, by_max)
        rx = rng.uniform(rx_min, rx_max)
        ry = rng.uniform(ry_min, ry_max)
        rq = rng.uniform(rq_min, rq_max)
        if (rx - bx) ** 2 + (ry - by) ** 2 < d_min ** 2:
            continue
        states.append((rx, ry, rq, bx, by))

    if len(states) < num_states:
        raise RuntimeError(
            f"Only sampled {len(states)}/{num_states} uniform validation states")

    return np.array(states[:num_states], dtype=np.float32)


def sample_states_for_generation(pop_size, num_trials, generation,
                                 total_generations, config, rng=None):
    """Sample initial states for one generation.

    Computes p from linear schedule and generates pop_size * num_trials states.

    Returns:
        (pop_size, num_trials, 5) array of (rx, ry, rq, bx, by)
    """
    if rng is None:
        rng = np.random.RandomState()

    challenge = config.get('challenge', {})
    schedule = challenge.get('schedule', 'linear')
    p_min = challenge.get('p_min', 0.022)
    total_gen = challenge.get('total_generations', total_generations)

    if schedule == 'linear':
        t = min(generation / max(total_gen - 1, 1), 1.0)
        p = p_min + (1.0 - p_min) * t
    elif schedule == 'quadratic':
        t = min(generation / max(total_gen - 1, 1), 1.0)
        p = p_min + (1.0 - p_min) * t ** 2
    else:
        p = challenge.get('value', 1.0)

    p = min(max(p, 0.0), 1.0)

    total = pop_size * num_trials
    states = sample_forward_l4_states(total, p, config, rng)
    return states.reshape(pop_size, num_trials, 5), p
