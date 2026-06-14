import importlib.util
import unittest
from pathlib import Path

import numpy as np
import mujoco


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class ActionSafeCollectionTest(unittest.TestCase):
    def test_next_cartesian_command_is_bounded_from_observed_pose(self):
        command = SyncSimRaccoonDataset.next_cartesian_command(
            observed_xyz=[0.0, 0.0, 0.0],
            target_xyz=[0.0, 0.0, 0.01],
            max_step=0.002,
        )

        np.testing.assert_allclose(command, [0.0, 0.0, 0.002])

    def test_cartesian_interpolation_respects_step_limit(self):
        points = SyncSimRaccoonDataset.interpolate_cartesian_segment(
            start_xyz=[0.0, 0.0, 0.0],
            target_xyz=[0.0, 0.0, 0.01],
            max_step=0.004,
        )

        self.assertEqual(len(points), 3)
        path = np.asarray([[0.0, 0.0, 0.0], *points], dtype=float)
        self.assertLessEqual(
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).max()),
            0.004 + 1e-12,
        )
        np.testing.assert_allclose(points[-1], [0.0, 0.0, 0.01])

    def test_pre_close_alignment_rejects_gripper_still_above_object(self):
        self.assertFalse(
            SyncSimRaccoonDataset.is_pre_close_aligned(
                observed_xyz=[0.0, 0.18, 0.05],
                target_xyz=[0.0, 0.18, 0.018],
                tolerance=0.012,
            )
        )

    def test_pre_close_alignment_accepts_gripper_at_grasp_height(self):
        self.assertTrue(
            SyncSimRaccoonDataset.is_pre_close_aligned(
                observed_xyz=[0.001, 0.181, 0.025],
                target_xyz=[0.0, 0.18, 0.018],
                tolerance=0.012,
            )
        )

    def test_recorded_transition_rejects_large_ee_delta(self):
        previous = {
            "ee_pose": [0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0, 0.0],
        }
        next_observation = {
            "ee_pose": [0.006, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0, 0.0],
        }

        with self.assertRaisesRegex(RuntimeError, "EE transition exceeds limit"):
            SyncSimRaccoonDataset.validate_recorded_transition(
                previous,
                next_observation,
                max_ee_delta=0.005,
            )

    def test_recorded_transition_accepts_bounded_motion(self):
        previous = {
            "ee_pose": [0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0, 0.0],
        }
        next_observation = {
            "ee_pose": [0.003, 0.0, 0.0],
            "joint_angles": [0.1, 0.0, 0.0, 0.0],
        }

        SyncSimRaccoonDataset.validate_recorded_transition(
            previous,
            next_observation,
            max_ee_delta=0.005,
            max_joint_delta=0.35,
        )

    def test_target_clearance_rejects_nearby_distractor(self):
        specs = SyncSimRaccoonDataset.make_default_object_specs()
        specs["red"]["x"], specs["red"]["y"] = 0.0, 0.18
        specs["blue"]["x"], specs["blue"]["y"] = 0.03, 0.18

        with self.assertRaisesRegex(ValueError, "target clearance violation"):
            SyncSimRaccoonDataset.validate_target_clearance(
                specs,
                target_color="red",
                min_clearance=0.07,
            )

    def test_non_target_motion_is_rejected(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        dataset.data = mujoco.MjData(dataset.model)
        specs = SyncSimRaccoonDataset.make_default_object_specs()
        mujoco.mj_resetData(dataset.model, dataset.data)
        dataset.reset_colored_objects(specs, target_color="red")
        mujoco.mj_forward(dataset.model, dataset.data)
        initial_positions = dataset.get_non_target_object_positions(specs, "red")

        body_name = specs["blue"]["body_name"]
        qpos_adr, _ = dataset._get_freejoint_addresses(body_name)
        dataset.data.qpos[qpos_adr] += 0.004
        mujoco.mj_forward(dataset.model, dataset.data)

        with self.assertRaisesRegex(RuntimeError, "non-target object moved"):
            dataset.validate_non_target_objects_undisturbed(
                specs,
                target_color="red",
                initial_positions=initial_positions,
                max_displacement=0.003,
            )

    def test_non_target_vertical_settling_is_not_counted_as_disturbance(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        dataset.data = mujoco.MjData(dataset.model)
        specs = SyncSimRaccoonDataset.make_default_object_specs()
        mujoco.mj_resetData(dataset.model, dataset.data)
        dataset.reset_colored_objects(specs, target_color="red")
        mujoco.mj_forward(dataset.model, dataset.data)
        initial_positions = dataset.get_non_target_object_positions(specs, "red")

        body_name = specs["blue"]["body_name"]
        qpos_adr, _ = dataset._get_freejoint_addresses(body_name)
        dataset.data.qpos[qpos_adr + 2] -= 0.006
        mujoco.mj_forward(dataset.model, dataset.data)

        dataset.validate_non_target_objects_undisturbed(
            specs,
            target_color="red",
            initial_positions=initial_positions,
            max_displacement=0.003,
        )


if __name__ == "__main__":
    unittest.main()
