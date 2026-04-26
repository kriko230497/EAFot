"""Pitch geometry — goal segments only (open mode).

In open mode, only the 6 goal-net segments are solid boundaries.
The field perimeter is permeable — ball leaving = trial ends.
"""

import jax.numpy as jnp


def build_goal_segments(hl, hw, hgw, gd):
    """Build the 6 goal-net segments as arrays.

    Returns (seg_x1, seg_y1, seg_dx, seg_dy, seg_len_sq) as plain lists.
    """
    segments = []

    # Left goal (3 segments): top post, back wall, bottom post
    segments.append((-hl, hgw, -hl - gd, hgw))       # top post
    segments.append((-hl - gd, hgw, -hl - gd, -hgw))  # back wall
    segments.append((-hl - gd, -hgw, -hl, -hgw))      # bottom post

    # Right goal (3 segments): bottom post, back wall, top post
    segments.append((hl, -hgw, hl + gd, -hgw))        # bottom post
    segments.append((hl + gd, -hgw, hl + gd, hgw))     # back wall
    segments.append((hl + gd, hgw, hl, hgw))           # top post

    seg_x1 = [s[0] for s in segments]
    seg_y1 = [s[1] for s in segments]
    seg_dx = [s[2] - s[0] for s in segments]
    seg_dy = [s[3] - s[1] for s in segments]
    seg_len_sq = [dx ** 2 + dy ** 2 for dx, dy in zip(seg_dx, seg_dy)]

    return seg_x1, seg_y1, seg_dx, seg_dy, seg_len_sq


def is_goal_scored(bx, by, ball_radius, params):
    """Check if ball has fully crossed a goal line.

    Returns: 1 (right goal), -1 (left goal), 0 (no goal).
    """
    in_goal_width = jnp.abs(by) <= params.half_goal_width
    right = (bx - ball_radius >= params.half_length) & in_goal_width
    left = (bx + ball_radius <= -params.half_length) & in_goal_width
    return jnp.where(right, 1, jnp.where(left, -1, 0))


def is_ball_out_of_play(bx, by, ball_radius, params):
    """Check if ball has left the field (open mode).

    Ball is out if it crosses a sideline or endline outside the goal.
    """
    hl = params.half_length
    hw = params.half_width
    hgw = params.half_goal_width

    out_top = by - ball_radius >= hw
    out_bottom = by + ball_radius <= -hw
    out_right = (bx - ball_radius >= hl) & (jnp.abs(by) > hgw)
    out_left = (bx + ball_radius <= -hl) & (jnp.abs(by) > hgw)

    return out_top | out_bottom | out_right | out_left


def ray_cast_multi(ox, oy, directions, params):
    """Cast multiple rays against goal segments. Returns (num_rays,) distances.

    Args:
        ox, oy: ray origin (scalars)
        directions: (num_rays,) angles in world frame
        params: StaticParams with segment arrays
    """
    dx = jnp.cos(directions)  # (R,)
    dy = jnp.sin(directions)  # (R,)

    # Broadcast: (R, 1) vs (1, S)
    dx_2d = dx[:, None]
    dy_2d = dy[:, None]
    seg_dx = params.seg_dx[None, :]
    seg_dy = params.seg_dy[None, :]

    fx = ox - params.seg_x1[None, :]  # (1, S) broadcast to (R, S)
    fy = oy - params.seg_y1[None, :]

    denom = dx_2d * seg_dy - dy_2d * seg_dx  # (R, S)
    parallel = jnp.abs(denom) < 1e-12
    safe_denom = jnp.where(parallel, 1.0, denom)

    t = (fx * seg_dy - fy * seg_dx) / safe_denom   # ray parameter
    u = (fx * dy_2d - fy * dx_2d) / safe_denom     # segment parameter

    valid = (~parallel) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    hits = jnp.where(valid, t, jnp.inf)

    return jnp.min(hits, axis=1)  # (R,)


def distance_to_segments(px, py, params):
    """Minimum distance from point to any goal segment."""
    fx = px - params.seg_x1
    fy = py - params.seg_y1
    len_sq = jnp.clip(params.seg_len_sq, 1e-12, None)
    t = jnp.clip((fx * params.seg_dx + fy * params.seg_dy) / len_sq, 0.0, 1.0)
    proj_x = params.seg_x1 + t * params.seg_dx
    proj_y = params.seg_y1 + t * params.seg_dy
    dists = jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return jnp.min(dists)
