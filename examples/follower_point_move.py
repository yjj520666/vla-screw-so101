"""SO-100/101 taught joint-point movement. Default operation never writes motors.

Uses the locally installed LeRobot Feetech bus API, not simulation joint values.
Requires explicit calibration and joint limits; never calibrates or enables torque.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def positions(value):
    if set(value) != set(JOINTS):
        raise ValueError('Expected exactly six named joints')
    result = {k: float(value[k]) for k in JOINTS}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Non-finite joint position')
    return result


def plan(start, target, limits, seconds=5., hz=20., speed=5., acceleration=10.):
    start, target = positions(start), positions(target)
    for value in (seconds, hz, speed, acceleration):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Timing and motion limits must be finite and positive')
    if hz > 50 or seconds > 120:
        raise ValueError('Maximum 50 Hz / 120 seconds')
    if abs(start['gripper'] - target['gripper']) > 0.01:
        raise ValueError('Point movement must preserve the current gripper position')
    for k in JOINTS:
        low, high = map(float, limits[k])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f'Invalid limits: {k}')
        if not low <= start[k] <= high or not low <= target[k] <= high:
            raise ValueError(f'Joint outside supplied limits: {k}')
    distance = max(abs(target[k] - start[k]) for k in JOINTS)
    if distance > 20:
        raise ValueError('First-stage movement limited to 20 degrees per joint')
    # Quintic minimum-jerk interpolation: peak s\'=1.875, peak |s\'\'|<5.774.
    seconds = max(seconds, 1.875 * distance / speed, math.sqrt(5.774 * distance / acceleration))
    if seconds > 120:
        raise ValueError('Requested limits require more than 120 seconds')
    count = math.ceil(seconds * hz)
    return [(seconds * i / count, {
        k: start[k] + (target[k] - start[k]) * (10*u**3 - 15*u**4 + 6*u**5)
        for k in JOINTS}) for i in range(count + 1) for u in [i / count]]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def make_bus(port, calibration):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    if set(calibration) != set(JOINTS):
        raise ValueError('Calibration must contain exactly six joints')
    for i, k in enumerate(JOINTS, 1):
        if calibration[k]['id'] != i:
            raise ValueError('Expected SO follower motor IDs 1..6')
    return FeetechMotorsBus(port=port, motors={
        k: Motor(i, 'sts3215', MotorNormMode.RANGE_0_100 if k == 'gripper' else MotorNormMode.DEGREES)
        for i, k in enumerate(JOINTS, 1)
    }, calibration={k: MotorCalibration(**v) for k, v in calibration.items()})


def run_motion(bus, trajectory, limits, log, tolerance=2.):
    previous = positions(bus.sync_read('Present_Position'))
    if max(abs(previous[k] - trajectory[0][1][k]) for k in JOINTS) > 0.5:
        raise RuntimeError('Arm moved after planning; re-plan')
    for register, expected in [('Torque_Enable', 1), ('Operating_Mode', 0)]:
        if any(int(v) != expected for v in bus.sync_read(register, normalize=False).values()):
            raise RuntimeError(f'{register} not ready; this tool will not change it')
    started = time.monotonic()
    # No catch-up bursts. A missed deadline aborts instead of jumping ahead.
    for timestamp, command in trajectory:
        time.sleep(max(0., started + timestamp - time.monotonic()))
        if time.monotonic() - started - timestamp > 0.15:
            raise RuntimeError('Control deadline missed')
        actual = positions(bus.sync_read('Present_Position'))
        if any(not limits[k][0] <= actual[k] <= limits[k][1] for k in JOINTS):
            raise RuntimeError('Measured joint outside limits')
        if max(abs(actual[k] - previous[k]) for k in JOINTS) > 5.:
            raise RuntimeError('Tracking error exceeds 5 degrees / gripper units')
        bus.sync_write('Goal_Position', command)
        previous = command
        log.write(json.dumps({'t': time.monotonic() - started, 'goal': command, 'actual': actual}) + '\n')
        log.flush()
    deadline = time.monotonic() + 3.
    stable = 0
    while time.monotonic() < deadline:
        actual = positions(bus.sync_read('Present_Position'))
        error = max(abs(actual[k] - previous[k]) for k in JOINTS)
        stable = stable + 1 if error <= tolerance else 0
        if error > 5:
            raise RuntimeError('Final tracking error exceeds limit')
        if stable >= 5:
            log.write(json.dumps({'success': True, 'max_error': error, 'actual': actual}) + '\n')
            return
        time.sleep(.05)
    raise RuntimeError('Target not reached within settling timeout')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['capture', 'move'])
    p.add_argument('--port', required=True)
    p.add_argument('--calibration', required=True, help='Explicit existing calibration JSON')
    p.add_argument('--pose', required=True, help='Captured pose JSON')
    p.add_argument('--limits', help='JSON joint-name: [min, max]; degrees, gripper 0..100')
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--execute', action='store_true', help='Actually send motor goals')
    p.add_argument('--output', default='follower_trajectory.jsonl')
    args = p.parse_args()
    if args.mode == 'capture' and args.execute:
        p.error('capture does not accept --execute')
    if args.mode == 'move' and not args.limits:
        p.error('move requires --limits')
    calibration = load(args.calibration)
    fingerprint = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    bus = make_bus(args.port, calibration)
    moving = False
    try:
        # Do NOT call Robot.connect(): it configures motors and can change torque.
        bus.connect()
        if not bus.is_calibrated:
            raise RuntimeError('Device calibration does not match the supplied file')
        start = positions(bus.sync_read('Present_Position'))
        if args.mode == 'capture':
            with open(args.pose, 'x', encoding='utf-8') as f:
                json.dump({'calibration_sha256': fingerprint, 'joints': start}, f, indent=2)
            print('Captured without changing torque or position:', args.pose)
            return
        pose = load(args.pose)
        if pose['calibration_sha256'] != fingerprint:
            raise ValueError('Pose belongs to another calibration')
        limits = {k: list(map(float, v)) for k, v in load(args.limits).items()}
        trajectory = plan(start, pose['joints'], limits, seconds=args.seconds)
        with open(args.output, 'x', encoding='utf-8') as log:
            if not args.execute:
                for t, q in trajectory:
                    log.write(json.dumps({'t': t, 'goal': q, 'dry_run': True}) + '\n')
                print('DRY RUN: trajectory exported; no motor writes:', args.output)
                return
            moving = True
            run_motion(bus, trajectory, limits, log)
            moving = False
            print('Target reached. Torque remains unchanged; support arm before powering off.')
    except BaseException:
        if moving:
            # Stop the trajectory; best-effort hold. Not a physical emergency stop.
            try:
                actual = positions(bus.sync_read('Present_Position'))
                bus.sync_write('Goal_Position', actual)
            except Exception:
                print('Hold failed: use the physical emergency stop / power switch.')
        raise
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == '__main__':
    main()
