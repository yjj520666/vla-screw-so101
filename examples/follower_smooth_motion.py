"""Continuous coordinated SO-101 joint motion for COM6.

The path uses one trapezoidal scalar profile for every joint: short acceleration,
constant-speed cruise, then deceleration. Commands are write-only during motion so
serial feedback latency cannot introduce a pause at every waypoint.
"""
from __future__ import annotations

import math
import time

from follower_point_move import JOINTS, positions


def trapezoid_duration(distance: float, speed: float, acceleration: float) -> tuple[float, float, float]:
    """Return acceleration time, cruise time and peak velocity for a scalar distance."""
    if not all(math.isfinite(v) and v > 0 for v in (speed, acceleration)):
        raise ValueError("speed and acceleration must be finite and positive")
    if distance < 0 or not math.isfinite(distance):
        raise ValueError("invalid distance")
    if distance == 0:
        return 0.0, 0.0, 0.0
    ramp = speed / acceleration
    ramp_distance = 0.5 * acceleration * ramp * ramp
    if 2 * ramp_distance >= distance:  # triangular profile for short moves
        ramp = math.sqrt(distance / acceleration)
        return ramp, 0.0, acceleration * ramp
    return ramp, (distance - 2 * ramp_distance) / speed, speed


def scalar_distance(t: float, distance: float, ramp: float, cruise: float, peak: float) -> float:
    total = 2 * ramp + cruise
    t = min(total, max(0.0, t))
    acceleration = peak / ramp if ramp else 0.0
    if t <= ramp:
        return 0.5 * acceleration * t * t
    ramp_distance = 0.5 * peak * ramp
    if t <= ramp + cruise:
        return ramp_distance + peak * (t - ramp)
    remaining = total - t
    return distance - 0.5 * acceleration * remaining * remaining


def build_trajectory(start, target, speed=12.0, acceleration=30.0, hz=50.0):
    """Build a coordinated, pause-free trajectory in calibrated degrees/percent."""
    start, target = positions(start), positions(target)
    if not math.isfinite(hz) or not 20 <= hz <= 100:
        raise ValueError("hz must be between 20 and 100")
    if abs(start["gripper"] - target["gripper"]) > 0.01:
        raise ValueError("smooth point motion preserves the gripper")
    distance = max(abs(target[k] - start[k]) for k in JOINTS if k != "gripper")
    ramp, cruise, peak = trapezoid_duration(distance, speed, acceleration)
    total = 2 * ramp + cruise
    count = max(1, math.ceil(total * hz))
    trajectory = []
    for i in range(count + 1):
        t = total * i / count
        fraction = scalar_distance(t, distance, ramp, cruise, peak) / distance if distance else 1.0
        trajectory.append((t, {k: start[k] + (target[k] - start[k]) * fraction for k in JOINTS}))
    return trajectory


def execute_write_only(bus, trajectory, limits, log=None, deadline_slack=0.015):
    """Send goals on an absolute clock; no serial reads occur between first and last goal."""
    current = positions(bus.sync_read("Present_Position"))
    if max(abs(current[k] - trajectory[0][1][k]) for k in JOINTS) > 0.75:
        raise RuntimeError("start pose changed; rebuild trajectory")
    for _, command in trajectory:
        for k, value in command.items():
            low, high = limits[k]
            if not low <= value <= high:
                raise ValueError(f"trajectory exceeds {k} limit")
    bus.sync_write("Goal_Position", current)
    if any(int(v) != 1 for v in bus.sync_read("Torque_Enable", normalize=False).values()):
        bus.enable_torque()
    started = time.perf_counter()
    missed = 0
    for timestamp, command in trajectory:
        deadline = started + timestamp
        time.sleep(max(0.0, deadline - time.perf_counter()))
        lateness = time.perf_counter() - deadline
        if lateness > deadline_slack:
            missed += 1
        bus.sync_write("Goal_Position", command)
        if log is not None:
            log.write(f'{timestamp:.6f},{lateness:.6f},{command}\n')
    final = positions(bus.sync_read("Present_Position"))
    return {"final": final, "missed_deadlines": missed, "duration": trajectory[-1][0]}
