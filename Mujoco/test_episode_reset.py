import importlib.util
import unittest
from pathlib import Path

import mujoco
import numpy as np


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class EpisodeResetTest(unittest.TestCase):
    def make_dataset_without_renderer(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        dataset.data = mujoco.MjData(dataset.model)
        dataset.current_setpoints = [0.0] * 5
        dataset.target_angles = [0.0] * 4
        dataset.joint_control_mode = [dataset.MODE_POSITION] * 4
        dataset.gripper_target = dataset.GRIP_OPEN
        dataset.gripper_mode = dataset.GRIP_MODE_FREE
        dataset.active_object_body_name = dataset.CYLINDER_BODY_BY_COLOR["red"]
        dataset.active_object_type = "cylinder"
        dataset.step_n = lambda _steps: None
        return dataset

    def test_reset_episode_clears_accumulated_mujoco_state(self):
        dataset = self.make_dataset_without_renderer()
        specs = dataset.make_default_object_specs()

        dataset.data.time = 12.5
        dataset.data.qvel[:] = 3.0
        dataset.data.qacc_warmstart[:] = 7.0
        dataset.data.qfrc_applied[:] = 5.0
        dataset.data.xfrc_applied[:] = 4.0
        dataset.data.ctrl[:] = -2.0

        dataset.reset_episode(specs, target_color="green")

        self.assertEqual(dataset.data.time, 0.0)
        np.testing.assert_allclose(dataset.data.qvel, 0.0)
        np.testing.assert_allclose(dataset.data.qacc_warmstart, 0.0)
        np.testing.assert_allclose(dataset.data.qfrc_applied, 0.0)
        np.testing.assert_allclose(dataset.data.xfrc_applied, 0.0)
        self.assertEqual(dataset.active_object_body_name, "target_object_green")

        home = np.radians([0.0, -10.0, -140.0, 60.0])
        np.testing.assert_allclose(dataset.data.qpos[:4], home)
        np.testing.assert_allclose(dataset.data.ctrl[:4], home)
        self.assertAlmostEqual(dataset.data.qpos[4], dataset.GRIP_OPEN)
        self.assertAlmostEqual(dataset.data.ctrl[4], dataset.GRIP_OPEN)
        self.assertFalse(dataset.grasp_stabilizer_active)
        self.assertIsNone(dataset.grasp_stabilizer_offset)
        self.assertIsNone(dataset.grasp_stabilizer_min_z)
        self.assertFalse(dataset.grasp_contact_confirmed)

        for color, spec in specs.items():
            body_id = mujoco.mj_name2id(
                dataset.model, mujoco.mjtObj.mjOBJ_BODY, spec["body_name"]
            )
            joint_id = int(dataset.model.body_jntadr[body_id])
            qpos_adr = int(dataset.model.jnt_qposadr[joint_id])
            np.testing.assert_allclose(
                dataset.data.qpos[qpos_adr:qpos_adr + 3],
                [spec["x"], spec["y"], dataset.OBJECT_GEOM_CONFIGS[spec["object_type"]]["z"]],
            )


if __name__ == "__main__":
    unittest.main()
