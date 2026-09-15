# VLA 打螺丝 · SO-101 控制项目

这是一个新建的独立大创项目仓库。当前可运行部分是 **SO-101 从臂 XYZ 定点控制**；视觉识别、语言指令理解、螺丝抓取/旋入和端到端 VLA 策略尚未集成，不能把控制界面当作完整的 VLA 打螺丝系统。

## 已有功能

- Windows 图形界面可自定义 COM 端口，显示编码器推算的实时 XYZ、关节角、温度、电压与力矩状态。
- 输入绝对 XYZ 或相对 dXYZ（mm），选择 5–60 °/s 的关节速度，预览 IK 后执行。
- 50 Hz 连续轨迹，加速、匀速、减速，中途不逐点驻停；可停止并保持当前位置。
- 基于设备标定、关节范围、位置模式、温度和电压进行检查。

## 运行

1. Windows / Python 3.10+；创建环境并安装 `requirements.txt`。`tkinter` 随标准 Windows Python 安装。
2. **把你自己的 SO-101 从臂标定文件**放到 `configs/so101_device_calibration.json`，或设置环境变量 `SO101_CALIBRATION_PATH` 指向它。标定文件不上传 GitHub；不能借用另一只臂的标定。
3. 运行 `start_ui.bat` 或 `python examples/so101_xyz_ui.py`，选择实际端口并连接。不会自动移动机械臂。
4. 命令行可运行 `python examples/so101_xyz_control.py --port COM6 --calibration <标定文件路径>`。

坐标原点是第一关节中心，Z 向上，X/Y 对应 MuJoCo 基座轴。Z 运动在原真机环境测试过；迁移到新环境时要重新验证 X/Y 方向和可达区。实时 XYZ 是**编码器 + 模型 FK 估计**，不是外部相机测量。

## 项目结构

- `examples/so101_xyz_ui.py`: GUI 和实时状态。
- `examples/so101_xyz_control.py`: FK/IK 与命令行控制。
- `examples/follower_point_move.py`: Feetech 总线与标定检查。
- `examples/follower_smooth_motion.py`: 连续轨迹。
- `mujoco/so101_kinematics.xml`: 无 STL 依赖的 SO-101 运动学模型；只用于 FK/IK，不作为打螺丝接触动力学模型。
- `docs/ROADMAP.md`: 大创下一阶段计划。

## 验证边界

离线单元测试验证端口解析和模型 IK。此前真机运动验证发生在原项目与原环境；本独立副本尚未重新连接或驱动硬件。
