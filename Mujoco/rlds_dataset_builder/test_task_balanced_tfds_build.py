import ast
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_DIR / "build_task_balanced_tfds.py"
INTERMEDIATE_BASE = (
    SCRIPT_DIR.parent / "raccoon_dataset" / "task_balanced_intermediate"
)


class TaskBalancedTfdsBuildTest(unittest.TestCase):
    def test_all_three_intermediate_manifests_exist(self):
        if not INTERMEDIATE_BASE.is_dir():
            self.skipTest(
                "Generated task-balanced intermediate data is intentionally "
                "excluded from Git."
            )

        expected = {
            "raccoon_grasp": (1080, 120),
            "raccoon_push": (146, 16),
            "raccoon_pick_and_place": (131, 15),
        }
        for dataset_name, expected_counts in expected.items():
            root = INTERMEDIATE_BASE / dataset_name
            counts = []
            for split in ("train", "val"):
                manifest = root / f"manifest_{split}.jsonl"
                self.assertTrue(manifest.is_file(), manifest)
                counts.append(
                    sum(
                        1
                        for line in manifest.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                )
            self.assertEqual(tuple(counts), expected_counts)

    def test_build_script_defines_three_distinct_builder_classes(self):
        tree = ast.parse(BUILD_SCRIPT.read_text(encoding="utf-8"))
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertTrue(
            {"RaccoonGrasp", "RaccoonPush", "RaccoonPickAndPlace"}
            <= class_names
        )


if __name__ == "__main__":
    unittest.main()
