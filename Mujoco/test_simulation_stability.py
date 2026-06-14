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


class SimulationStabilityTest(unittest.TestCase):
    def setUp(self):
        self.dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        self.dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        self.dataset.data = mujoco.MjData(self.dataset.model)

    def test_clean_state_is_stable(self):
        self.assertIsNone(self.dataset.get_simulation_instability_reason())

    def test_non_finite_acceleration_is_rejected(self):
        self.dataset.data.qacc[0] = np.nan

        self.assertEqual(
            self.dataset.get_simulation_instability_reason(),
            "non-finite qacc",
        )

    def test_robot_and_gripper_joints_have_passive_stabilization(self):
        robot_dofs = self.dataset.model.dof_damping[:10]
        robot_armature = self.dataset.model.dof_armature[:10]

        self.assertTrue(np.all(robot_dofs > 0.0))
        self.assertTrue(np.all(robot_armature > 0.0))


if __name__ == "__main__":
    unittest.main()
