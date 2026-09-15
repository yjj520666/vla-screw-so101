import unittest
import numpy as np

from follower_point_move import JOINTS
from so101_xyz_control import Kinematics, DEFAULT_MODEL


class XYZTests(unittest.TestCase):
    def test_current_position_round_trip(self):
        current = {
            "shoulder_pan": -3.956043956,
            "shoulder_lift": -103.868131868,
            "elbow_flex": 95.824175824,
            "wrist_flex": -101.758241758,
            "wrist_roll": -2.021978022,
            "gripper": 30.740568235,
        }
        limits = {name: [-170.0, 170.0] for name in JOINTS}
        limits["gripper"] = [0.0, 100.0]
        kinematics = Kinematics(DEFAULT_MODEL)
        xyz = kinematics.xyz_mm(current)
        solved, error = kinematics.inverse(current, xyz, limits)
        self.assertLess(error, 1.0)
        self.assertLess(max(abs(solved[k] - current[k]) for k in JOINTS), 0.1)

    def test_ten_mm_up(self):
        current = {
            "shoulder_pan": -3.956043956,
            "shoulder_lift": -103.868131868,
            "elbow_flex": 95.824175824,
            "wrist_flex": -101.758241758,
            "wrist_roll": -2.021978022,
            "gripper": 30.740568235,
        }
        limits = {name: [-105.0, 105.0] for name in JOINTS}
        limits["wrist_roll"] = [-160.0, 160.0]
        limits["gripper"] = [0.0, 100.0]
        kinematics = Kinematics(DEFAULT_MODEL)
        xyz = kinematics.xyz_mm(current)
        solved, error = kinematics.inverse(current, xyz + np.array([0.0, 0.0, 10.0]), limits)
        self.assertLess(error, 1.0)
        self.assertEqual(solved["wrist_roll"], current["wrist_roll"])
        self.assertEqual(solved["gripper"], current["gripper"])


if __name__ == "__main__":
    unittest.main()
