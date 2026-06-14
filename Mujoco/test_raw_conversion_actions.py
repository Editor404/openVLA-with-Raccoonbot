import json
import tempfile
import unittest
from pathlib import Path

from raccoon_dataset.convert_raw_to_openvla_rlds_intermediate import convert_episode


class RawConversionActionTests(unittest.TestCase):
    def test_idle_filter_keeps_immediate_raw_frame_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_episode = root / "episode_000001"
            out_episode = root / "converted" / "episode_000001"
            raw_episode.mkdir(parents=True)

            ee_x = [0.0, 0.005, 0.010, 0.015, 0.020]
            joints = [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0],
            ]
            steps = []
            for index, (x, joint_angles) in enumerate(zip(ee_x, joints)):
                image_file = f"frame_{index:06d}.png"
                (raw_episode / image_file).write_bytes(b"test-image")
                steps.append(
                    {
                        "t": index,
                        "image_file": image_file,
                        "joint_angles": joint_angles,
                        "gripper_state": 0.0,
                        "object_pose": [0.0, 0.0, 0.0, 0.0],
                        "ee_pose": [x, 0.0, 0.0],
                        "action": [0.0, 0.0, 0.0, 0.0],
                    }
                )

            meta = {
                "episode_id": 1,
                "instruction": "grasp the red cylinder",
                "task_type": "grasp",
                "target_color": "red",
                "target_object_type": "cylinder",
                "target_body_name": "target_object",
                "dataset_generator_version": "test-version",
                "collection_config": {
                    "grasp_stabilizer_enabled": False,
                    "grasp_mode": "mujoco_physics_only",
                },
                "success": True,
                "goal_xy": [0.0, 0.18],
                "box_init_xy": [0.0, 0.18],
                "box_init_yaw": 0.0,
                "steps": steps,
            }
            (raw_episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            convert_episode(
                raw_episode_dir=raw_episode,
                out_episode_dir=out_episode,
                drop_idle_steps=True,
                min_joint_delta_norm=0.05,
                min_ee_delta_norm=0.01,
            )

            converted = json.loads((out_episode / "episode.json").read_text(encoding="utf-8"))
            converted_steps = converted["steps"]

            self.assertEqual([step["raw_index"] for step in converted_steps], [0, 3, 4])
            self.assertAlmostEqual(converted_steps[0]["action"][0], 0.005)
            self.assertNotAlmostEqual(converted_steps[0]["action"][0], 0.015)
            self.assertAlmostEqual(converted_steps[1]["action"][0], 0.005)
            self.assertEqual(converted_steps[2]["action"][:3], [0.0, 0.0, 0.0])
            metadata = converted["episode_metadata"]
            self.assertEqual(metadata["target_color"], "red")
            self.assertEqual(metadata["target_object_type"], "cylinder")
            self.assertEqual(metadata["target_body_name"], "target_object")
            self.assertEqual(metadata["dataset_generator_version"], "test-version")
            self.assertEqual(
                metadata["collection_config"]["grasp_mode"],
                "mujoco_physics_only",
            )


if __name__ == "__main__":
    unittest.main()
