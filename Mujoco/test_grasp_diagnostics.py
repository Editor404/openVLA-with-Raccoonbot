import importlib.util
import unittest
from pathlib import Path

import mujoco


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class GraspDiagnosticsTest(unittest.TestCase):
    def test_diagnostic_contains_failure_inputs(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.model = mujoco.MjModel.from_xml_path(
            str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml")
        )
        dataset.data = mujoco.MjData(dataset.model)
        mujoco.mj_forward(dataset.model, dataset.data)

        diagnostic = dataset.get_grasp_diagnostic(
            target_body_name="target_object",
            stage="grasp_close",
            waypoint_index=2,
        )

        expected_keys = {
            "stage",
            "waypoint_index",
            "simulation_time",
            "touch_left",
            "touch_right",
            "touch_left_pass",
            "touch_right_pass",
            "gripper_qpos",
            "gripper_closed_pass",
            "target_z",
            "target_robot_contact_pass",
            "left_finger_contact_pass",
            "right_finger_contact_pass",
            "grasp_stabilizer_active",
            "grasp_contact_confirmed",
            "grasp_contact_source",
            "lift_delta",
            "lift_pass",
            "contacts",
        }
        self.assertEqual(set(diagnostic), expected_keys)
        self.assertEqual(diagnostic["stage"], "grasp_close")
        self.assertIsInstance(diagnostic["contacts"], list)


if __name__ == "__main__":
    unittest.main()
