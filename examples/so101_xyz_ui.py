"""Tkinter UI for continuous SO-101 XYZ control with live encoder feedback."""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from serial.tools import list_ports

import numpy as np

from follower_point_move import JOINTS, load, make_bus, positions
from follower_smooth_motion import build_trajectory
from so101_xyz_control import DEFAULT_CALIBRATION, DEFAULT_MODEL, DEFAULT_OUTPUT, Kinematics, calibrated_limits

PORT_SETTINGS = DEFAULT_OUTPUT / "ui_port_settings.json"


def validate_port(value: str) -> str:
    port = value.strip().upper()
    if not re.fullmatch(r"COM[1-9][0-9]*", port):
        raise ValueError("串口格式应为 COM4、COM6 等")
    return port


def load_saved_port() -> str:
    try:
        return validate_port(json.loads(PORT_SETTINGS.read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return "COM6"


def save_port(port: str) -> None:
    PORT_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    PORT_SETTINGS.write_text(json.dumps({"port": validate_port(port)}, indent=2), encoding="utf-8")


class RobotController:
    def __init__(self, events: queue.Queue):
        self.events = events
        self.calibration = load(DEFAULT_CALIBRATION)
        self.limits = calibrated_limits(self.calibration)
        self.kinematics = Kinematics(DEFAULT_MODEL)
        self.bus = None
        self.bus_lock = threading.Lock()
        self.kinematics_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.feedback_thread = None
        self.motion_thread = None
        self.connected = False
        self.port = None
        self.connecting = False
        self.moving = False
        self.torque_enabled = False
        self.latest_joints = None
        self.latest_xyz = None

    def post(self, kind: str, **payload) -> None:
        self.events.put((kind, payload))

    def connect(self, port: str) -> None:
        if self.connected or self.connecting:
            return
        self.connecting = True
        try:
            port = validate_port(port)
            bus = make_bus(port, self.calibration)
            bus.connect()
            if not bus.is_calibrated:
                raise RuntimeError(f"{port} 的电机标定与当前 SO-101 标定文件不匹配；不能把另一只臂的标定用于运动")
            if any(int(v) != 0 for v in bus.sync_read("Operating_Mode", normalize=False).values()):
                raise RuntimeError("部分电机不在位置模式")
            self.bus = bus
            self.connected = True
            self.port = port
            save_port(port)
            self.shutdown_event.clear()
            self.feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
            self.feedback_thread.start()
            self.post("connected", port=port)
        except Exception as error:
            try:
                if "bus" in locals() and bus.is_connected:
                    bus.disconnect(disable_torque=False)
            except Exception:
                pass
            self.post("error", message=f"{port if 'port' in locals() else '串口'} 连接失败：{error}")
        finally:
            self.connecting = False

    def disconnect(self, disable_torque: bool = False) -> None:
        self.shutdown_event.set()
        self.stop_event.set()
        if self.motion_thread and self.motion_thread.is_alive():
            self.motion_thread.join(timeout=2.0)
        if self.feedback_thread and self.feedback_thread.is_alive():
            self.feedback_thread.join(timeout=1.0)
        with self.bus_lock:
            if self.bus and self.bus.is_connected:
                if disable_torque and self.torque_enabled:
                    self.bus.disable_torque()
                    self.torque_enabled = False
                self.bus.disconnect(disable_torque=False)
        self.connected = False
        self.bus = None
        self.port = None
        self.post("disconnected")

    def _feedback_loop(self) -> None:
        diagnostics_at = 0.0
        while not self.shutdown_event.is_set() and self.connected:
            started = time.perf_counter()
            try:
                with self.bus_lock:
                    joints = positions(self.bus.sync_read("Present_Position"))
                    torque = self.bus.sync_read("Torque_Enable", normalize=False)
                    temperature = voltage = None
                    if started >= diagnostics_at:
                        temperature = self.bus.sync_read("Present_Temperature", normalize=False)
                        voltage = self.bus.sync_read("Present_Voltage", normalize=False)
                        diagnostics_at = started + 1.0
                with self.kinematics_lock:
                    xyz = self.kinematics.xyz_mm(joints)
                self.latest_joints, self.latest_xyz = joints, xyz
                self.torque_enabled = all(int(value) == 1 for value in torque.values())
                self.post(
                    "feedback", joints=joints, xyz=xyz.tolist(), torque=self.torque_enabled,
                    temperature=temperature, voltage=voltage,
                )
            except Exception as error:
                self.post("error", message=f"实时读取失败：{error}")
                break
            time.sleep(max(0.0, 0.1 - (time.perf_counter() - started)))  # actual encoder data at 10 Hz

    def preview(self, values: np.ndarray, relative: bool, speed: float) -> None:
        if not self.connected:
            self.post("error", message="请先选择端口并连接")
            return
        threading.Thread(target=self._prepare, args=(values, relative, speed, False), daemon=True).start()

    def move(self, values: np.ndarray, relative: bool, speed: float) -> None:
        if not self.connected:
            self.post("error", message="请先选择端口并连接")
            return
        if self.moving:
            self.post("error", message="机械臂正在运动")
            return
        self.stop_event.clear()
        self.motion_thread = threading.Thread(
            target=self._prepare, args=(values, relative, speed, True), daemon=True
        )
        self.motion_thread.start()

    def _prepare(self, values: np.ndarray, relative: bool, speed: float, execute: bool) -> None:
        try:
            with self.bus_lock:
                current = positions(self.bus.sync_read("Present_Position"))
                temperature = self.bus.sync_read("Present_Temperature", normalize=False)
                voltage = self.bus.sync_read("Present_Voltage", normalize=False)
            if max(temperature.values()) >= 60 or min(voltage.values()) < 105:
                raise RuntimeError("温度或电压预检失败")
            with self.kinematics_lock:
                current_xyz = self.kinematics.xyz_mm(current)
                target_xyz = current_xyz + values if relative else values
                target_joints, ik_error = self.kinematics.inverse(current, target_xyz, self.limits)
            trajectory = build_trajectory(
                current, target_joints, speed=speed, acceleration=2.5 * speed, hz=50.0
            )
            self.post(
                "preview", current_xyz=current_xyz.tolist(), target_xyz=target_xyz.tolist(),
                target_joints=target_joints, ik_error=ik_error, duration=trajectory[-1][0], speed=speed,
            )
            if execute:
                self._execute(current, target_xyz, target_joints, trajectory, speed)
        except Exception as error:
            self.moving = False
            self.post("error", message=f"规划失败：{error}")

    def _execute(self, start, target_xyz, target_joints, trajectory, speed) -> None:
        self.moving = True
        self.post("motion_start", duration=trajectory[-1][0])
        missed = 0
        stopped = False
        started = time.perf_counter()
        try:
            with self.bus_lock:
                actual = positions(self.bus.sync_read("Present_Position"))
                if max(abs(actual[k] - start[k]) for k in JOINTS) > 0.75:
                    raise RuntimeError("起始姿态已变化，请重新规划")
                self.bus.sync_write("Goal_Position", actual)
                torque = self.bus.sync_read("Torque_Enable", normalize=False)
                if any(int(value) != 1 for value in torque.values()):
                    self.bus.enable_torque()
                self.torque_enabled = True
            for timestamp, command in trajectory:
                if self.stop_event.is_set():
                    stopped = True
                    break
                deadline = started + timestamp
                time.sleep(max(0.0, deadline - time.perf_counter()))
                lateness = time.perf_counter() - deadline
                missed += int(lateness > 0.015)
                with self.bus_lock:
                    self.bus.sync_write("Goal_Position", command)
                self.post("progress", value=timestamp / trajectory[-1][0] if trajectory[-1][0] else 1.0)
            if stopped:
                with self.bus_lock:
                    final = positions(self.bus.sync_read("Present_Position"))
                    self.bus.sync_write("Goal_Position", final)
                self.post("stopped", final=final)
            else:
                time.sleep(0.4)
                with self.bus_lock:
                    final = positions(self.bus.sync_read("Present_Position"))
                    temperature = self.bus.sync_read("Present_Temperature", normalize=False)
                    voltage = self.bus.sync_read("Present_Voltage", normalize=False)
                with self.kinematics_lock:
                    final_xyz = self.kinematics.xyz_mm(final)
                report = {
                    "requested_xyz_mm": target_xyz.tolist(), "final_xyz_mm": final_xyz.tolist(),
                    "cartesian_error_mm": float(np.linalg.norm(final_xyz - target_xyz)),
                    "speed_deg_s": speed, "duration_s": trajectory[-1][0],
                    "missed_deadlines": missed, "start_joints": start,
                    "target_joints": target_joints, "final_joints": final,
                    "temperature": temperature, "voltage": voltage,
                }
                DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
                path = DEFAULT_OUTPUT / f"ui_xyz_{time.strftime('%Y%m%d_%H%M%S')}.json"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                self.post("motion_done", report=report, path=str(path))
        except Exception as error:
            try:
                with self.bus_lock:
                    hold = positions(self.bus.sync_read("Present_Position"))
                    self.bus.sync_write("Goal_Position", hold)
            except Exception:
                pass
            self.post("error", message=f"运动中止并保持：{error}")
        finally:
            self.moving = False
            self.stop_event.clear()

    def hold(self) -> None:
        if not self.connected:
            self.post("error", message="请先连接端口")
            return
        try:
            with self.bus_lock:
                current = positions(self.bus.sync_read("Present_Position"))
                self.bus.sync_write("Goal_Position", current)
                self.bus.enable_torque()
            self.torque_enabled = True
            self.post("log", message="已保持当前位置")
        except Exception as error:
            self.post("error", message=f"保持失败：{error}")

    def torque_off(self) -> None:
        if not self.connected:
            self.post("error", message="请先连接端口")
            return
        if self.moving:
            self.post("error", message="运动中不能解除力矩，请先停止")
            return
        try:
            with self.bus_lock:
                self.bus.disable_torque()
            self.torque_enabled = False
            self.post("log", message="力矩已解除")
        except Exception as error:
            self.post("error", message=f"解除力矩失败：{error}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SO-101 XYZ 连续运动控制器")
        self.geometry("940x690")
        self.minsize(880, 640)
        self.events = queue.Queue()
        self.controller = RobotController(self.events)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.after(50, self._drain_events)
        self.refresh_ports()

    def _build(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Big.TLabel", font=("Microsoft YaHei UI", 23, "bold"))
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x")
        self.status = tk.StringVar(value="请选择串口并连接")
        ttk.Label(top, textvariable=self.status, style="Header.TLabel").pack(side="left")
        self.port_var = tk.StringVar(value=load_saved_port())
        ttk.Button(top, text="刷新串口", command=self.refresh_ports).pack(side="right", padx=4)
        ttk.Button(top, text="连接", command=self.connect).pack(side="right", padx=4)
        ttk.Button(top, text="断开", command=self.disconnect).pack(side="right", padx=4)
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, width=10)
        self.port_combo.pack(side="right", padx=4)
        ttk.Label(top, text="连接端口：").pack(side="right")

        xyz_box = ttk.LabelFrame(root, text="实时当前位置（编码器 10 Hz）", padding=12)
        xyz_box.pack(fill="x", pady=(12, 8))
        self.xyz_text = tk.StringVar(value="X ---   Y ---   Z --- mm")
        ttk.Label(xyz_box, textvariable=self.xyz_text, style="Big.TLabel").pack(anchor="center")
        self.diag_text = tk.StringVar(value="温度 --  电压 --  力矩 --")
        ttk.Label(xyz_box, textvariable=self.diag_text).pack(anchor="center", pady=(6, 0))

        middle = ttk.Frame(root)
        middle.pack(fill="x", pady=6)
        target = ttk.LabelFrame(middle, text="目标输入", padding=12)
        target.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.mode = tk.StringVar(value="relative")
        ttk.Radiobutton(target, text="相对位移 dX dY dZ", variable=self.mode, value="relative").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(target, text="绝对坐标 X Y Z", variable=self.mode, value="absolute").grid(row=1, column=0, columnspan=3, sticky="w")
        self.entries = []
        for column, name in enumerate(("X / dX", "Y / dY", "Z / dZ")):
            ttk.Label(target, text=name + " (mm)").grid(row=2, column=column, padx=5, pady=(12, 3))
            entry = ttk.Entry(target, width=14, justify="center")
            entry.insert(0, "0")
            entry.grid(row=3, column=column, padx=5)
            self.entries.append(entry)

        speed_box = ttk.LabelFrame(middle, text="速度", padding=12)
        speed_box.pack(side="right", fill="both", expand=True, padx=(6, 0))
        self.speed = tk.DoubleVar(value=30.0)
        self.speed_label = tk.StringVar(value="30 °/s")
        ttk.Label(speed_box, textvariable=self.speed_label, style="Header.TLabel").pack()
        ttk.Scale(speed_box, from_=5, to=60, variable=self.speed, command=lambda value: self.speed_label.set(f"{float(value):.0f} °/s")).pack(fill="x", pady=8)
        presets = ttk.Frame(speed_box)
        presets.pack()
        for value in (18, 30, 45, 60):
            ttk.Button(presets, text=str(value), width=5, command=lambda v=value: self._set_speed(v)).pack(side="left", padx=3)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=8)
        self.preview_button = ttk.Button(buttons, text="仅计算 / 预览", command=self.preview)
        self.preview_button.pack(side="left", padx=4)
        self.move_button = ttk.Button(buttons, text="执行连续运动", command=self.move)
        self.move_button.pack(side="left", padx=4)
        ttk.Button(buttons, text="停止并保持", command=self.controller.stop_event.set).pack(side="left", padx=4)
        ttk.Button(buttons, text="保持当前位置", command=self.controller.hold).pack(side="right", padx=4)
        ttk.Button(buttons, text="解除力矩", command=self.controller.torque_off).pack(side="right", padx=4)

        self.progress = ttk.Progressbar(root, maximum=100)
        self.progress.pack(fill="x", pady=(2, 8))

        joints_box = ttk.LabelFrame(root, text="实时关节角", padding=8)
        joints_box.pack(fill="x")
        self.joint_vars = {}
        for index, name in enumerate(JOINTS):
            ttk.Label(joints_box, text=name).grid(row=0, column=index, padx=7)
            variable = tk.StringVar(value="---")
            self.joint_vars[name] = variable
            ttk.Label(joints_box, textvariable=variable).grid(row=1, column=index, padx=7)

        log_box = ttk.LabelFrame(root, text="状态与记录", padding=6)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_box, height=8, state="disabled", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

    def _set_speed(self, value):
        self.speed.set(value)
        self.speed_label.set(f"{value} °/s")

    def refresh_ports(self):
        ports = sorted({item.device.upper() for item in list_ports.comports() if item.device.upper().startswith("COM")})
        self.port_combo["values"] = ports
        if not self.port_var.get() and ports:
            self.port_var.set(ports[0])
        self.append_log("检测到串口：" + (", ".join(ports) if ports else "无"))

    def connect(self):
        try:
            port = validate_port(self.port_var.get())
        except ValueError as error:
            messagebox.showerror("串口错误", str(error))
            return
        if self.controller.connected:
            if self.controller.port != port:
                messagebox.showinfo("切换串口", "请先断开当前串口，再选择新串口连接。")
            return
        if self.controller.connecting:
            return
        self.status.set(f"正在连接 {port}…")
        threading.Thread(target=self.controller.connect, args=(port,), daemon=True).start()

    def values(self):
        values = np.array([float(entry.get()) for entry in self.entries], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("XYZ 必须是有限数值")
        return values

    def preview(self):
        try:
            self.controller.preview(self.values(), self.mode.get() == "relative", float(self.speed.get()))
        except ValueError as error:
            messagebox.showerror("输入错误", str(error))

    def move(self):
        try:
            values = self.values()
        except ValueError as error:
            messagebox.showerror("输入错误", str(error))
            return
        mode = "相对位移" if self.mode.get() == "relative" else "绝对坐标"
        if not messagebox.askyesno("确认执行", f"{mode}: {values.tolist()} mm\n速度: {self.speed.get():.0f}°/s\n\n确认运动空间已清空？"):
            return
        self.controller.move(values, self.mode.get() == "relative", float(self.speed.get()))

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S ") + text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "connected":
                    self.status.set(f"{payload['port']} 已连接")
                    self.port_var.set(payload["port"])
                    self.port_combo.configure(state="disabled")
                    self.append_log(f"{payload['port']} 连接及标定检查通过")
                elif kind == "disconnected":
                    self.status.set("已断开")
                    self.port_combo.configure(state="normal")
                elif kind == "feedback":
                    xyz = payload["xyz"]
                    self.xyz_text.set(f"X {xyz[0]:8.1f}   Y {xyz[1]:8.1f}   Z {xyz[2]:8.1f} mm")
                    for name, value in payload["joints"].items():
                        self.joint_vars[name].set(f"{value:.1f}")
                    temp, volt = payload.get("temperature"), payload.get("voltage")
                    if temp and volt:
                        self.diag_text.set(f"最高温度 {max(temp.values())}°C   最低电压 {min(volt.values())/10:.1f}V   力矩 {'开启' if payload['torque'] else '关闭'}")
                elif kind == "preview":
                    self.append_log(f"规划：目标 {np.round(payload['target_xyz'],1).tolist()} mm，IK误差 {payload['ik_error']:.2f} mm，预计 {payload['duration']:.2f}s，速度 {payload['speed']:.0f}°/s")
                elif kind == "motion_start":
                    self.status.set("连续运动中…")
                    self.progress["value"] = 0
                elif kind == "progress":
                    self.progress["value"] = 100 * payload["value"]
                elif kind == "motion_done":
                    report = payload["report"]
                    self.status.set("目标到达，保持力矩")
                    self.progress["value"] = 100
                    self.append_log(f"完成：XYZ误差 {report['cartesian_error_mm']:.1f} mm；超时 {report['missed_deadlines']}；{payload['path']}")
                elif kind == "stopped":
                    self.status.set("已停止并保持")
                    self.append_log("用户停止，当前位置保持力矩")
                elif kind == "log":
                    self.append_log(payload["message"])
                elif kind == "error":
                    self.status.set("发生错误")
                    self.append_log(payload["message"])
                    messagebox.showerror("SO-101", payload["message"])
        except queue.Empty:
            pass
        self.after(50, self._drain_events)

    def disconnect(self):
        disable = self.controller.torque_enabled and messagebox.askyesno("断开", "断开时解除力矩？")
        threading.Thread(target=self.controller.disconnect, args=(disable,), daemon=True).start()

    def on_close(self):
        disable = self.controller.torque_enabled and messagebox.askyesno("退出", "退出时解除力矩？\n选择“否”将保持当前位置力矩。")
        self.controller.disconnect(disable)
        self.destroy()


if __name__ == "__main__":
    # COM ports are exclusive. Prevent a second desktop launch from competing.
    import ctypes
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\SO101_XYZ_UI")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        notice = tk.Tk()
        notice.withdraw()
        messagebox.showinfo("SO-101 XYZ", "控制软件已经在运行，请切换到现有窗口。")
        notice.destroy()
    else:
        App().mainloop()
