import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MUJOCO_DIR = Path(__file__).resolve().parent
CLIENT_DIR = MUJOCO_DIR.parents[1] / "executeCode"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "raccoon_dataset_collector",
    MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py",
)
client_module = load_module("raccoon_client_env", CLIENT_DIR / "raccoon_env.py")


class EePoseConsistencyTest(unittest.TestCase):
    def test_dataset_and_client_use_identical_gripper_tip_fk(self):
        joint_samples = (
            (0.0, -10.0, -140.0),
            (35.0, -30.0, -90.0),
            (-50.0, 15.0, -120.0),
        )

        for degrees in joint_samples:
            qpos = np.zeros(5, dtype=np.float64)
            qpos[:3] = np.radians(degrees)

            dataset_env = dataset_module.SyncSimRaccoonDataset.__new__(
                dataset_module.SyncSimRaccoonDataset
            )
            dataset_env.data = SimpleNamespace(qpos=qpos.copy())

            client_env = client_module.SyncSimRaccoonEnv.__new__(
                client_module.SyncSimRaccoonEnv
            )
            client_env.data = SimpleNamespace(qpos=qpos.copy())

            self.assertTrue(
                np.allclose(
                    dataset_env.get_ee_pose(),
                    client_env.get_ee_pose(),
                    atol=1e-12,
                ),
                msg=f"EE FK mismatch at joint angles {degrees}",
            )


if __name__ == "__main__":
    unittest.main()
