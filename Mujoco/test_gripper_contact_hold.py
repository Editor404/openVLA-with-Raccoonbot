import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class GripperContactHoldTest(unittest.TestCase):
    def test_gripper_close_speed_is_contact_safe(self):
        self.assertLessEqual(SyncSimRaccoonDataset.GRIPPER_SPEED, 3.0)

    def make_dataset(self, object_type="cube", qpos=-0.40):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.data = SimpleNamespace(qpos=np.array([0, 0, 0, 0, qpos], dtype=float))
        dataset.active_object_body_name = "target_object"
        dataset.active_object_type = object_type
        dataset.gripper_target = dataset.GRIP_CLOSE
        dataset.gripper_close_requested = True
        dataset.gripper_contact_hold_target = None
        dataset.grasp_stabilizer_active = False
        dataset.grasp_stabilizer_offset = None
        dataset.grasp_stabilizer_min_z = None
        dataset.grasp_contact_confirmed = False
        dataset.grasp_contact_source = None
        dataset.enable_grasp_stabilizer = False
        dataset.get_gripper_touch_state = lambda: (0.0, 0.0)
        dataset.get_target_finger_contact_state = lambda _: (True, True)
        dataset._activate_grasp_stabilizer = lambda: True
        return dataset

    def test_bilateral_finger_contact_latches_once_without_ratcheting(self):
        dataset = self.make_dataset(object_type="cube", qpos=-0.40)

        self.assertTrue(dataset._latch_gripper_on_contact())
        first_target = dataset.gripper_target
        self.assertAlmostEqual(first_target, -0.412)

        dataset.data.qpos[4] = -0.45
        self.assertTrue(dataset._latch_gripper_on_contact())
        self.assertEqual(dataset.gripper_target, first_target)
        self.assertTrue(dataset.grasp_contact_confirmed)
        self.assertEqual(dataset.grasp_contact_source, "target_bilateral_finger")
        self.assertFalse(dataset.grasp_stabilizer_active)

    def test_shape_specific_squeeze_is_smallest_for_sphere(self):
        cylinder = self.make_dataset(object_type="cylinder")
        sphere = self.make_dataset(object_type="sphere")

        cylinder._latch_gripper_on_contact()
        sphere._latch_gripper_on_contact()

        self.assertLess(cylinder.gripper_target, sphere.gripper_target)

    def test_pad_sensor_contact_without_target_contact_does_not_latch(self):
        dataset = self.make_dataset()
        dataset.get_gripper_touch_state = lambda: (3.0, 4.0)
        dataset.get_target_finger_contact_state = lambda _: (False, False)

        self.assertFalse(dataset._latch_gripper_on_contact())
        self.assertIsNone(dataset.gripper_contact_hold_target)
        self.assertEqual(dataset.gripper_target, dataset.GRIP_CLOSE)
        self.assertFalse(dataset.grasp_stabilizer_active)

    def test_unilateral_target_contact_does_not_latch(self):
        dataset = self.make_dataset()
        dataset.get_target_finger_contact_state = lambda _: (True, False)

        self.assertFalse(dataset._latch_gripper_on_contact())
        self.assertIsNone(dataset.gripper_contact_hold_target)
        self.assertFalse(dataset.grasp_stabilizer_active)

    def test_early_transient_contact_does_not_latch(self):
        dataset = self.make_dataset(object_type="cube", qpos=-0.05)

        self.assertFalse(dataset._latch_gripper_on_contact())
        self.assertIsNone(dataset.gripper_contact_hold_target)
        self.assertFalse(dataset.grasp_contact_confirmed)

    def test_open_resets_contact_latch(self):
        dataset = self.make_dataset()
        dataset._latch_gripper_on_contact()

        dataset.open_gripper()

        self.assertFalse(dataset.gripper_close_requested)
        self.assertIsNone(dataset.gripper_contact_hold_target)
        self.assertFalse(dataset.grasp_stabilizer_active)
        self.assertFalse(dataset.grasp_contact_confirmed)
        self.assertEqual(dataset.gripper_target, dataset.GRIP_OPEN)

    def test_stabilizer_moves_contacted_object_with_end_effector_delta(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        dataset.data = mujoco.MjData(dataset.model)
        mujoco.mj_forward(dataset.model, dataset.data)
        dataset.active_object_body_name = "target_object"
        dataset.grasp_stabilizer_active = False
        dataset.grasp_stabilizer_offset = None
        dataset.grasp_stabilizer_min_z = None
        dataset.grasp_contact_confirmed = False
        dataset.grasp_contact_source = None
        dataset.enable_grasp_stabilizer = True
        dataset.get_target_finger_contact_state = lambda _: (True, True)

        pad_midpoint = np.array([0.0, 0.0, 0.0], dtype=float)
        dataset.get_gripper_pad_midpoint = lambda: pad_midpoint.copy()
        self.assertTrue(dataset._activate_grasp_stabilizer())
        self.assertEqual(
            dataset.grasp_contact_source,
            "target_bilateral_finger",
        )
        before = dataset.get_object_pose("target_object")[:3].copy()

        pad_midpoint[:] = before + [0.01, -0.02, 0.03]
        dataset._apply_grasp_stabilizer()
        after = dataset.get_object_pose("target_object")[:3]

        np.testing.assert_allclose(
            after - before,
            [0.01, -0.02, 0.03],
            atol=1e-6,
        )

    def test_stabilizer_rejects_direct_activation_without_bilateral_contact(self):
        dataset = self.make_dataset()
        dataset.get_target_finger_contact_state = lambda _: (False, False)

        self.assertFalse(
            SyncSimRaccoonDataset._activate_grasp_stabilizer(dataset)
        )
        self.assertFalse(dataset.grasp_stabilizer_active)
        self.assertFalse(dataset.grasp_contact_confirmed)
        self.assertIsNone(dataset.grasp_contact_source)

    def test_success_rejects_lift_when_contact_is_not_retained_at_close(self):
        dataset = self.make_dataset(qpos=-0.50)
        dataset.get_object_pose = lambda _: np.array([0.0, 0.0, 0.06, 0.0])

        self.assertFalse(
            dataset.is_target_grasp_success(
                target_body_name="target_object",
                initial_target_z=0.01,
                grasp_contact_confirmed=True,
                grasp_contact_retained=False,
            )
        )

    def test_success_accepts_confirmed_retained_contact_and_lift(self):
        dataset = self.make_dataset(qpos=-0.50)
        dataset.get_object_pose = lambda _: np.array([0.0, 0.0, 0.06, 0.0])

        self.assertTrue(
            dataset.is_target_grasp_success(
                target_body_name="target_object",
                initial_target_z=0.01,
                grasp_contact_confirmed=True,
                grasp_contact_retained=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
