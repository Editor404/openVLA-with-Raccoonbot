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


class ObjectColorMappingTest(unittest.TestCase):
    def test_xml_body_and_rgba_match_semantic_colors(self):
        model = mujoco.MjModel.from_xml_path(str(MUJOCO_DIR / "Raccoon_colored_cylinder.xml"))
        SyncSimRaccoonDataset.validate_scene_color_mapping(model)

    def test_generated_specs_keep_canonical_color_body_mapping(self):
        for seed in range(100):
            specs = SyncSimRaccoonDataset.sample_object_specs(np.random.default_rng(seed))
            SyncSimRaccoonDataset.validate_object_specs(specs)
            for color, body_name in SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.items():
                self.assertEqual(specs[color]["body_name"], body_name)

    def test_mismatched_color_body_mapping_is_rejected(self):
        specs = SyncSimRaccoonDataset.make_default_object_specs()
        specs["red"]["body_name"], specs["blue"]["body_name"] = (
            specs["blue"]["body_name"],
            specs["red"]["body_name"],
        )

        with self.assertRaisesRegex(ValueError, "object_specs mapping error"):
            SyncSimRaccoonDataset.validate_object_specs(specs)

    def test_placement_order_is_randomized_and_recorded(self):
        first_colors = set()
        for seed in range(100):
            specs = SyncSimRaccoonDataset.sample_object_specs(np.random.default_rng(seed))
            ranks = {color: spec["placement_rank"] for color, spec in specs.items()}
            self.assertEqual(sorted(ranks.values()), list(range(len(specs))))
            first_colors.add(min(ranks, key=ranks.get))

        self.assertEqual(first_colors, set(SyncSimRaccoonDataset.CYLINDER_COLORS))

    def test_forced_target_clearance_is_enforced_during_sampling(self):
        for seed in range(50):
            specs = SyncSimRaccoonDataset.sample_object_specs(
                np.random.default_rng(seed),
                object_types=("cylinder", "cube", "sphere"),
                forced_target={"color": "green", "object_type": "cube"},
                target_clearance=0.07,
            )

            self.assertEqual(specs["green"]["placement_rank"], 0)
            self.assertEqual(specs["green"]["object_type"], "cube")
            SyncSimRaccoonDataset.validate_target_clearance(
                specs,
                target_color="green",
                min_clearance=0.07,
            )


if __name__ == "__main__":
    unittest.main()
