import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_multitask_colored_objects_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_multitask_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultitaskColoredObjectsTest(unittest.TestCase):
    def setUp(self):
        self.dataset = MODULE.SyncSimRaccoonDataset.__new__(
            MODULE.SyncSimRaccoonDataset
        )

    def test_default_count_is_thirty_per_task_color_object(self):
        units = tuple(
            (task, color, object_type)
            for task in ("push", "pick_and_place")
            for color in ("red", "blue", "green", "yellow")
            for object_type in ("cylinder", "cube", "sphere")
        )
        counts = MODULE._balanced_target_counts(720, units)
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {30})

    def test_push_plan_closes_before_contact_approach(self):
        plan = self.dataset.make_push_plan(
            object_x=0.0,
            object_y=0.17,
            goal_xy=(0.0, 0.21),
            object_type="cube",
        )
        self.assertEqual([step[3] for step in plan], [0, 1, 1, 1, 0])
        self.assertEqual(plan[-2][:2], [0.0, 0.21])

    def test_pick_and_place_closes_only_after_reaching_object(self):
        plan = self.dataset.make_pick_and_place_plan(
            object_x=0.0,
            object_y=0.18,
            goal_xy=(0.09, 0.11),
            object_type="sphere",
        )
        self.assertEqual([step[3] for step in plan], [0, 0, 1, 1, 1, 1, 0, 0])
        self.assertEqual(plan[1][:3], plan[2][:3])
        self.assertEqual(plan[6][:2], [0.09, 0.11])

    def test_task_plans_are_ik_valid_for_center_workspace(self):
        for task_type, goal_xy in (
            ("push", (0.0, 0.21)),
            ("pick_and_place", (0.09, 0.11)),
        ):
            plan = self.dataset.validate_task_plan_ik(
                task_type,
                object_x=0.0,
                object_y=0.17,
                goal_xy=goal_xy,
                object_type="cube",
            )
            self.assertGreaterEqual(len(plan), 4)

    def test_forced_push_target_uses_center_corridor(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            specs = MODULE.SyncSimRaccoonDataset.sample_object_specs(
                rng=rng,
                colors=("red",),
                object_types=("cylinder", "cube", "sphere"),
                forced_target={
                    "color": "red",
                    "object_type": "cylinder",
                    "x_range": (-0.010, 0.010),
                    "y_range": (0.145, 0.180),
                },
                x_range=(-0.12, 0.12),
                y_range=(0.14, 0.22),
            )
            self.assertLessEqual(abs(specs["red"]["x"]), 0.010)
            self.assertGreaterEqual(specs["red"]["y"], 0.145)
            self.assertLessEqual(specs["red"]["y"], 0.180)
            goal = MODULE._sample_task_goal("push", specs["red"], specs)
            self.assertLessEqual(goal[1], 0.220)

    def test_resume_recovers_counts_and_next_episode_id(self):
        units = (
            ("push", "green", "cylinder"),
            ("pick_and_place", "red", "cube"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episodes = (
                (1, True, "push", "green", "cylinder"),
                (3, True, "pick_and_place", "red", "cube"),
                (5, False, "push", "green", "cylinder"),
            )
            for episode_id, success, task, color, object_type in episodes:
                episode_dir = root / f"episode_{episode_id:06d}"
                episode_dir.mkdir()
                with open(episode_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "episode_id": episode_id,
                            "success": success,
                            "task_type": task,
                            "target_color": color,
                            "target_object_type": object_type,
                        },
                        f,
                    )

            counts, next_id, loaded, skipped = (
                MODULE._load_existing_multitask_progress(root, units)
            )
            self.assertEqual(counts[("push", "green", "cylinder")], 1)
            self.assertEqual(counts[("pick_and_place", "red", "cube")], 1)
            self.assertEqual(next_id, 6)
            self.assertEqual(loaded, 2)
            self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
