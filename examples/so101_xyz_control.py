"""Interactive XYZ point controller for the calibrated COM6 SO-101 follower arm."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MUJOCO_SITE = ROOT / ".venv-mujoco-render" / "Lib" / "site-packages"
if str(MUJOCO_SITE) not in sys.path:
    sys.path.append(str(MUJOCO_SITE))

import mujoco
import numpy as np

from follower_point_move import JOINTS, load, make_bus, positions
from follower_smooth_motion import build_trajectory, execute_write_only

ARM_JOINTS = JOINTS[:4]
DEFAULT_CALIBRATION = Path(os.environ.get(
    "SO101_CALIBRATION_PATH", str(ROOT / "configs" / "so101_device_calibration.json")
))
DEFAULT_MODEL = ROOT / "mujoco" / "so101_kinematics.xml"
DEFAULT_OUTPUT = ROOT / "outputs" / "follower_xyz"


def hardware_to_model(q: dict[str, float]) -> np.ndarray:
    """Map calibrated LeRobot degrees to the verified MuJoCo joint convention."""
    return np.array([
        math.radians(q["shoulder_pan"]),
        math.radians(90.0 - q["shoulder_lift"]),
        math.radians(q["elbow_flex"] + 90.0),
        math.radians(q["wrist_flex"]),
        math.radians(q["wrist_roll"]),
        q["gripper"] / 100.0,
    ])


def model_to_hardware(q: np.ndarray, gripper: float) -> dict[str, float]:
    return {
        "shoulder_pan": math.degrees(q[0]),
        "shoulder_lift": 90.0 - math.degrees(q[1]),
        "elbow_flex": math.degrees(q[2]) - 90.0,
        "wrist_flex": math.degrees(q[3]),
        "wrist_roll": math.degrees(q[4]),
        "gripper": gripper,
    }


class Kinematics:
    """Numerical FK/IK using the same SO-101 geometry validated by the Z-axis tests."""

    def __init__(self, model_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.tcp_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1_fixed_jaw")
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot1_rotation")
        # Approximate midpoint between the two fingertip locations in the local fixed-jaw frame.
        self.tcp_offset = np.array([0.0, -0.085, 0.0])

    def _set(self, model_q: np.ndarray) -> None:
        self.data.qpos[:6] = model_q
        mujoco.mj_forward(self.model, self.data)

    def tcp_world(self, model_q: np.ndarray) -> np.ndarray:
        self._set(model_q)
        rotation = self.data.xmat[self.tcp_body].reshape(3, 3)
        return self.data.xpos[self.tcp_body] + rotation @ self.tcp_offset

    def base_origin(self, model_q: np.ndarray) -> np.ndarray:
        self._set(model_q)
        return self.data.xpos[self.base_body].copy()

    def xyz_mm(self, hardware_q: dict[str, float]) -> np.ndarray:
        q = hardware_to_model(hardware_q)
        return (self.tcp_world(q) - self.base_origin(q)) * 1000.0

    def _point_jacobian(self, model_q: np.ndarray) -> np.ndarray:
        self._set(model_q)
        rotation = self.data.xmat[self.tcp_body].reshape(3, 3)
        point = self.data.xpos[self.tcp_body] + rotation @ self.tcp_offset
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.tcp_body)
        return jacp[:, :4]

    def inverse(
        self,
        current: dict[str, float],
        target_xyz_mm: np.ndarray,
        limits: dict[str, list[float]],
        tolerance_mm: float = 1.0,
    ) -> tuple[dict[str, float], float]:
        q = hardware_to_model(current)
        base = self.base_origin(q)
        target_world = base + target_xyz_mm / 1000.0
        start_world = self.tcp_world(q)
        if np.linalg.norm(target_world - start_world) > 0.30:
            raise ValueError("single command is limited to 300 mm")

        # Damped least squares, staying on the current IK branch. Wrist roll and gripper remain fixed.
        for _ in range(500):
            point = self.tcp_world(q)
            error = target_world - point
            if np.linalg.norm(error) * 1000.0 <= tolerance_mm:
                break
            jacobian = self._point_jacobian(q)
            damping = 2e-4
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error)
            norm = np.linalg.norm(step)
            if norm > 0.04:
                step *= 0.04 / norm
            q[:4] += step
            candidate = model_to_hardware(q, current["gripper"])
            for index, name in enumerate(ARM_JOINTS):
                low, high = limits[name]
                candidate[name] = min(high, max(low, candidate[name]))
            q = hardware_to_model(candidate)
        final_error = float(np.linalg.norm(self.tcp_world(q) - target_world) * 1000.0)
        if final_error > tolerance_mm:
            raise ValueError(f"target is unreachable on the current IK branch (error {final_error:.1f} mm)")
        target = model_to_hardware(q, current["gripper"])
        target["wrist_roll"] = current["wrist_roll"]
        target["gripper"] = current["gripper"]
        return positions(target), final_error


def calibrated_limits(calibration: dict, margin_degrees: float = 1.0) -> dict[str, list[float]]:
    result = {}
    for name in JOINTS:
        if name == "gripper":
            result[name] = [0.0, 100.0]
        else:
            half = (calibration[name]["range_max"] - calibration[name]["range_min"]) * 180.0 / 4095.0
            result[name] = [-half + margin_degrees, half - margin_degrees]
    return result


def read_xyz(prompt: str) -> np.ndarray:
    text = input(prompt).strip().replace(",", " ")
    values = [float(value) for value in text.split()]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("请输入三个有限数值，例如：0 0 250")
    return np.asarray(values, dtype=float)


def choose_speed() -> float:
    text = input("速度 °/s [18/30/45/60，默认30]：").strip()
    speed = 30.0 if not text else float(text)
    if not 5.0 <= speed <= 60.0:
        raise ValueError("速度范围为 5–60°/s")
    return speed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    calibration = load(args.calibration)
    limits = calibrated_limits(calibration)
    kinematics = Kinematics(args.model)
    bus = make_bus(args.port, calibration)
    torque_enabled = False
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        bus.connect()
        if not bus.is_calibrated:
            raise RuntimeError(f"{args.port} 标定参数与配置文件不匹配")
        if any(int(v) != 0 for v in bus.sync_read("Operating_Mode", normalize=False).values()):
            raise RuntimeError("部分电机不在位置模式")
        print("\nSO-101 XYZ 连续运动控制器；单位 mm，速度单位 °/s")
        print("坐标系：原点=第一关节中心，X/Y=模型基座平面轴，Z=竖直向上")
        print("首次测试 X/Y 方向时请使用不超过 10 mm 的相对位移。")
        while True:
            current = positions(bus.sync_read("Present_Position"))
            xyz = kinematics.xyz_mm(current)
            print(f"\n当前位置 XYZ = {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f} mm")
            mode = input("输入 a=绝对坐标，r=相对位移，q=退出：").strip().lower()
            if mode == "q":
                break
            if mode not in {"a", "r"}:
                print("无效选项")
                continue
            try:
                entered = read_xyz("输入 X Y Z：" if mode == "a" else "输入 dX dY dZ：")
                target_xyz = entered if mode == "a" else xyz + entered
                speed = choose_speed()
                target_joints, ik_error = kinematics.inverse(current, target_xyz, limits)
                trajectory = build_trajectory(current, target_joints, speed=speed, acceleration=2.5 * speed, hz=50.0)
                print(f"目标 XYZ={target_xyz.round(1).tolist()} mm")
                print(f"IK误差={ik_error:.2f} mm，预计时间={trajectory[-1][0]:.2f} s")
                print("目标关节：" + ", ".join(f"{k}={target_joints[k]:.1f}" for k in JOINTS[:5]))
                if input("输入 y 执行，其余键取消：").strip().lower() != "y":
                    print("已取消")
                    continue
                pre_temperature = bus.sync_read("Present_Temperature", normalize=False)
                pre_voltage = bus.sync_read("Present_Voltage", normalize=False)
                if max(pre_temperature.values()) >= 60 or min(pre_voltage.values()) < 105:
                    raise RuntimeError("温度或电压预检失败")
                result = execute_write_only(bus, trajectory, limits)
                torque_enabled = True
                time.sleep(0.4)
                final = positions(bus.sync_read("Present_Position"))
                final_xyz = kinematics.xyz_mm(final)
                report = {
                    "requested_xyz_mm": target_xyz.tolist(),
                    "final_xyz_mm": final_xyz.tolist(),
                    "cartesian_error_mm": float(np.linalg.norm(final_xyz - target_xyz)),
                    "speed_deg_s": speed,
                    "duration_s": result["duration"],
                    "missed_deadlines": result["missed_deadlines"],
                    "start_joints": current,
                    "target_joints": target_joints,
                    "final_joints": final,
                }
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path = DEFAULT_OUTPUT / f"interactive_xyz_{timestamp}.json"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"完成：XYZ={final_xyz.round(1).tolist()} mm，误差={report['cartesian_error_mm']:.1f} mm")
                print("记录：", path)
            except (ValueError, RuntimeError) as error:
                print("拒绝执行：", error)
    except KeyboardInterrupt:
        print("\n已中断；尝试保持当前位置。")
        if bus.is_connected:
            try:
                hold = positions(bus.sync_read("Present_Position"))
                bus.sync_write("Goal_Position", hold)
                torque_enabled = all(int(v) == 1 for v in bus.sync_read("Torque_Enable", normalize=False).values())
            except Exception:
                print("保持失败，请使用硬件断电开关。")
    finally:
        if bus.is_connected:
            if torque_enabled:
                answer = input("退出时解除力矩？输入 y 解除，否则保持：").strip().lower()
                if answer == "y":
                    bus.disable_torque()
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
