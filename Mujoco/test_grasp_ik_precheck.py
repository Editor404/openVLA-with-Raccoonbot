import importlib.util
import unittest
from pathlib import Path


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class GraspIkPrecheckTest(unittest.TestCase):
    def setUp(self):
        self.dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)

    def test_reachable_target_returns_complete_plan(self):
        plan = self.dataset.validate_grasp_plan_ik(0.0, 0.18)

        self.assertEqual(plan, self.dataset.make_grasp_plan(0.0, 0.18))
        self.assertEqual(len(plan), 4)

    def test_low_grasp_waypoint_is_rejected_before_execution(self):
        with self.assertRaisesRegex(
            ValueError,
            r"grasp IK precheck failed: waypoint=1 xyz_m=\(-0.1000, 0.2500, 0.0200\)",
        ):
            self.dataset.validate_grasp_plan_ik(-0.10, 0.25)

    def test_duplicate_close_waypoint_is_only_checked_once(self):
        calls = []
        original_ik = self.dataset._calc_inv_kinematics

        def recording_ik(x, y, z):
            calls.append((x, y, z))
            return original_ik(x, y, z)

        self.dataset._calc_inv_kinematics = recording_ik
        self.dataset.validate_grasp_plan_ik(0.0, 0.18)

        self.assertEqual(len(calls), 3)

    def test_grasp_height_is_shape_aware(self):
        cube_plan = self.dataset.make_grasp_plan(0.0, 0.18, object_type="cube")
        sphere_plan = self.dataset.make_grasp_plan(0.0, 0.18, object_type="sphere")
        cylinder_plan = self.dataset.make_grasp_plan(0.0, 0.18, object_type="cylinder")

        self.assertEqual(cube_plan[1][2], 0.018)
        self.assertEqual(sphere_plan[1][2], 0.018)
        self.assertEqual(cylinder_plan[1][2], 0.020)


if __name__ == "__main__":
    unittest.main()
