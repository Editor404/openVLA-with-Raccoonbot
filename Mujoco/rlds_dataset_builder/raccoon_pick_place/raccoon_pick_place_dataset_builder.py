from typing import Iterator, Tuple, Any
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INTERMEDIATE_ROOT = (
    SCRIPT_DIR.parent.parent
    / "raccoon_dataset"
    / "task_balanced_intermediate"
    / "raccoon_pick_and_place"
)


class RaccoonPickPlace(tfds.core.GeneratorBasedBuilder):
    """TFDS/RLDS builder for Raccoon pick-and-place dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release for single-task Raccoon grasp dataset.",
    }
    INTERMEDIATE_ROOT = DEFAULT_INTERMEDIATE_ROOT
    DATASET_DESCRIPTION = "TFDS/RLDS dataset for RaccoonBot demonstrations."

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Put the converted intermediate dataset under INTERMEDIATE_ROOT.

    Supported layouts:

    Layout A:
      INTERMEDIATE_ROOT/
        manifest_train.jsonl
        manifest_val.jsonl
        episode_000001/
          episode.json
          images/
            frame_000000.png
            ...

    Layout B:
      INTERMEDIATE_ROOT/
        train/
          episode_000001/
            episode.json
            images/
              frame_000000.png
              ...
        val/
          episode_000101/
            episode.json
            images/
              frame_000000.png
              ...
    """

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.DATASET_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Main camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="Robot state: [q1,q2,q3,q4,0,0,0,gripper].",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc="EEF_POS action: [dx,dy,dz,droll,dpitch,dyaw,gripper_cmd]. "
                            "Rotation deltas are zero-filled because raw data does not include EE orientation.",
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount. Default 1.0 for demonstrations.",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Reward. 1.0 on final successful step, else 0.0.",
                    ),
                    "is_first": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on first step of the episode.",
                    ),
                    "is_last": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode.",
                    ),
                    "is_terminal": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step if terminal.",
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction for the episode.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_id": tfds.features.Scalar(dtype=np.int32),
                    "success": tfds.features.Scalar(dtype=np.bool_),
                    "goal_xy": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                    "box_init_xy": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                    "box_init_yaw": tfds.features.Scalar(dtype=np.float32),
                    "source_path": tfds.features.Text(
                        doc="Path to the source episode directory.",
                    ),
                }),
            }),
            supervised_keys=None,
            homepage="local",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root = Path(self.INTERMEDIATE_ROOT)

        train_manifest = root / "manifest_train.jsonl"
        val_manifest = root / "manifest_val.jsonl"

        has_manifest_layout = train_manifest.exists()
        has_dir_layout = (root / "train").exists()

        if has_manifest_layout:
            splits = {
                "train": self._generate_examples(source=str(train_manifest), mode="manifest"),
            }
            if val_manifest.exists():
                splits["val"] = self._generate_examples(source=str(val_manifest), mode="manifest")
            return splits

        if has_dir_layout:
            splits = {
                "train": self._generate_examples(source=str(root / "train"), mode="dir"),
            }
            if (root / "val").exists():
                splits["val"] = self._generate_examples(source=str(root / "val"), mode="dir")
            return splits

        raise FileNotFoundError(
            f"Could not find a supported dataset layout under: {root}\n"
            f"Expected either manifest_train.jsonl / manifest_val.jsonl or train/ val/ directories."
        )

    def _generate_examples(self, source: str, mode: str):
        source_path = Path(source)

        if mode == "manifest":
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)

                    if "episode_dir" in item:
                        episode_dir = Path(item["episode_dir"])
                    elif "path" in item:
                        episode_dir = Path(item["path"])
                    elif "relative_episode_json" in item:
                        episode_json_path = Path(self.INTERMEDIATE_ROOT) / item["relative_episode_json"]
                        episode_dir = episode_json_path.parent
                    elif "raw_episode_dir" in item and "split" in item:
                        episode_dir = (
                            Path(self.INTERMEDIATE_ROOT)
                            / item["split"]
                            / item["raw_episode_dir"]
                        )
                    else:
                        raise KeyError(f"Unsupported manifest format: {item}")

                    parsed = self._parse_episode_dir(episode_dir)
                    if parsed is not None:
                        yield parsed

        elif mode == "dir":
            episode_dirs = sorted([p for p in source_path.glob("episode_*") if p.is_dir()])
            for episode_dir in episode_dirs:
                parsed = self._parse_episode_dir(episode_dir)
                if parsed is not None:
                    yield parsed
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _parse_episode_dir(self, episode_dir: Path):
        episode_json_path = episode_dir / "episode.json"

        if not episode_json_path.exists():
            raw_meta_path = episode_dir / "meta.json"
            if raw_meta_path.exists():
                return self._parse_raw_meta_episode(episode_dir, raw_meta_path)
            raise FileNotFoundError(
                f"Neither episode.json nor meta.json found in {episode_dir}"
            )

        with open(episode_json_path, "r", encoding="utf-8") as f:
            ep = json.load(f)

        episode_metadata = ep.get("episode_metadata", {})
        steps_in = ep.get("steps", [])

        if len(steps_in) == 0:
            return None

        steps = []
        for i, step in enumerate(steps_in):
            obs = step.get("observation", {})

            image_path = self._resolve_image_path_from_intermediate(episode_dir, obs, step)
            state = self._ensure_float32_vector(obs.get("state", []), expected_dim=8)
            action = self._ensure_float32_vector(step.get("action", []), expected_dim=7)

            language_instruction = step.get(
                "language_instruction",
                ep.get("instruction", episode_metadata.get("instruction", "")),
            )

            is_first = bool(step.get("is_first", i == 0))
            is_last = bool(step.get("is_last", i == len(steps_in) - 1))
            is_terminal = bool(step.get("is_terminal", is_last))

            reward = step.get("reward", None)
            if reward is None:
                reward = 1.0 if (is_last and bool(episode_metadata.get("success", False))) else 0.0

            discount = step.get("discount", 1.0)

            steps.append({
                "observation": {
                    "image": str(image_path),
                    "state": state,
                },
                "action": action,
                "discount": np.float32(discount),
                "reward": np.float32(reward),
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
                "language_instruction": str(language_instruction),
            })

        sample = {
            "steps": steps,
            "episode_metadata": {
                "episode_id": int(episode_metadata.get("episode_id", -1)),
                "success": bool(episode_metadata.get("success", False)),
                "goal_xy": self._ensure_float32_vector(
                    episode_metadata.get("goal_xy", [0.0, 0.0]), expected_dim=2
                ),
                "box_init_xy": self._ensure_float32_vector(
                    episode_metadata.get("box_init_xy", [0.0, 0.0]), expected_dim=2
                ),
                "box_init_yaw": np.float32(episode_metadata.get("box_init_yaw", 0.0)),
                "source_path": str(episode_dir),
            },
        }

        return str(episode_dir), sample

    def _parse_raw_meta_episode(self, episode_dir: Path, meta_path: Path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_steps = meta.get("steps", [])
        if len(raw_steps) < 2:
            return None

        instruction = str(meta.get("instruction", ""))
        success = bool(meta.get("success", False))

        steps = []
        for i in range(len(raw_steps)):
            cur = raw_steps[i]
            nxt = raw_steps[i + 1] if i + 1 < len(raw_steps) else raw_steps[i]

            image_path = self._resolve_image_path_from_raw(episode_dir, cur)
            state = self._build_state_from_raw(cur)
            action = self._build_action_from_raw(cur, nxt)

            is_first = bool(cur.get("is_first", i == 0))
            is_last = bool(cur.get("is_last", i == len(raw_steps) - 1))
            is_terminal = is_last
            reward = 1.0 if (is_last and success) else 0.0

            steps.append({
                "observation": {
                    "image": str(image_path),
                    "state": state,
                },
                "action": action,
                "discount": np.float32(1.0),
                "reward": np.float32(reward),
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
                "language_instruction": instruction,
            })

        sample = {
            "steps": steps,
            "episode_metadata": {
                "episode_id": int(meta.get("episode_id", -1)),
                "success": success,
                "goal_xy": self._ensure_float32_vector(meta.get("goal_xy", [0.0, 0.0]), expected_dim=2),
                "box_init_xy": self._ensure_float32_vector(meta.get("box_init_xy", [0.0, 0.0]), expected_dim=2),
                "box_init_yaw": np.float32(meta.get("box_init_yaw", 0.0)),
                "source_path": str(episode_dir),
            },
        }

        return str(episode_dir), sample

    @staticmethod
    def _resolve_image_path_from_intermediate(episode_dir: Path, obs: dict, step: dict) -> Path:
        candidates = []

        if "image_path" in obs:
            candidates.append(episode_dir / obs["image_path"])
        if "image_file" in obs:
            candidates.append(episode_dir / obs["image_file"])
            candidates.append(episode_dir / "images" / obs["image_file"])
        if "image" in obs and isinstance(obs["image"], str):
            candidates.append(episode_dir / obs["image"])
            candidates.append(episode_dir / "images" / obs["image"])

        if "image_path" in step:
            candidates.append(episode_dir / step["image_path"])
        if "image_file" in step:
            candidates.append(episode_dir / step["image_file"])
            candidates.append(episode_dir / "images" / step["image_file"])
        if "image" in step and isinstance(step["image"], str):
            candidates.append(episode_dir / step["image"])
            candidates.append(episode_dir / "images" / step["image"])

        seen = set()
        uniq_candidates = []
        for c in candidates:
            c = c.resolve()
            if c not in seen:
                uniq_candidates.append(c)
                seen.add(c)

        for c in uniq_candidates:
            if c.exists():
                return c

        image_dir = episode_dir / "images"
        if image_dir.exists():
            pngs = sorted(image_dir.glob("*.png"))
            if len(pngs) > 0:
                return pngs[0]

        raise FileNotFoundError(
            f"Could not resolve image path in intermediate episode: {episode_dir}\n"
            f"Tried candidates:\n" + "\n".join(str(x) for x in uniq_candidates)
        )

    @staticmethod
    def _resolve_image_path_from_raw(episode_dir: Path, step: dict) -> Path:
        if "image_file" not in step:
            raise KeyError(f"'image_file' missing in raw step under {episode_dir}")
        image_path = episode_dir / step["image_file"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    @staticmethod
    def _build_state_from_raw(step: dict) -> np.ndarray:
        joint_angles = step.get("joint_angles", [])
        gripper_state = float(step.get("gripper_state", 0.0))

        if len(joint_angles) != 4:
            raise ValueError(f"Expected 4 joint angles, got {len(joint_angles)}")

        return np.array(
            [joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3], 0.0, 0.0, 0.0, gripper_state],
            dtype=np.float32,
        )

    @staticmethod
    def _build_action_from_raw(cur_step: dict, next_step: dict) -> np.ndarray:
        cur_ee = cur_step.get("ee_pose", [])
        nxt_ee = next_step.get("ee_pose", [])

        if len(cur_ee) < 3 or len(nxt_ee) < 3:
            raise ValueError("Expected ee_pose with at least 3 values [x, y, z] in both current and next steps")

        dpos = np.asarray(nxt_ee[:3], dtype=np.float32) - np.asarray(cur_ee[:3], dtype=np.float32)

        raw_action = cur_step.get("action", [0.0, 0.0, 0.0, 0.0])
        gripper_cmd = float(raw_action[3]) if len(raw_action) >= 4 else 0.0

        return np.array(
            [dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0, gripper_cmd],
            dtype=np.float32,
        )

    @staticmethod
    def _ensure_float32_vector(x, expected_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"Expected vector dim {expected_dim}, got {arr.shape[0]}")
        return arr
