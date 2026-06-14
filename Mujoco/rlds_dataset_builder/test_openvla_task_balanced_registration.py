import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OXE_ROOT = (
    REPO_ROOT
    / "openvla"
    / "prismatic"
    / "vla"
    / "datasets"
    / "rlds"
    / "oxe"
)


class OpenVlaTaskBalancedRegistrationTest(unittest.TestCase):
    def test_oxe_files_are_syntactically_valid(self):
        for name in ("configs.py", "transforms.py", "mixtures.py"):
            ast.parse((OXE_ROOT / name).read_text(encoding="utf-8"))

    def test_equal_weight_mixture_is_registered(self):
        text = (OXE_ROOT / "mixtures.py").read_text(encoding="utf-8")
        self.assertIn('"raccoon_task_balanced"', text)
        for dataset_name in (
            "raccoon_grasp",
            "raccoon_push",
            "raccoon_pick_and_place",
        ):
            self.assertIn(f'("{dataset_name}", 1.0)', text)

    def test_gripper_conversion_is_registered_for_all_tasks(self):
        text = (OXE_ROOT / "transforms.py").read_text(encoding="utf-8")
        self.assertIn("1.0 - tf.clip_by_value", text)
        for dataset_name in (
            "raccoon_grasp",
            "raccoon_push",
            "raccoon_pick_and_place",
        ):
            self.assertIn(
                f'"{dataset_name}": raccoon_colored_objects_transform',
                text,
            )


if __name__ == "__main__":
    unittest.main()
