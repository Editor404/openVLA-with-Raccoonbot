from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Raw episode folder  ->  OpenVLA-friendly RLDS intermediate
# ============================================================
# Input (raw)
#   raw_root/
#     episode_000001/
#       frame_000000.png
#       frame_000001.png
#       ...
#       meta.json
#
# Output (intermediate)
#   out_root/
#     dataset_info.json
#     manifest_train.jsonl
#     manifest_val.jsonl               # only if val_ratio > 0
#     train/
#       episode_000001/
#         images/
#           frame_000000.png
#           ...
#         episode.json
#     val/
#       episode_000123/
#         images/
#         episode.json
#
# episode.json schema (builder-friendly, RLDS-like)
# {
#   "episode_metadata": {...},
#   "steps": [
#      {
#        "observation": {
#           "image": "images/frame_000000.png",
#           "state": [8 dims],
#           ... debug fields ...
#        },
#        "action": [7 dims],
#        "language_instruction": "...",
#        "reward": 0.0,
#        "discount": 1.0,
#        "is_first": true,
#        "is_last": false,
#        "is_terminal": false,
#        "timestep": 0,
#        ... raw/debug fields ...
#      }
#   ]
# }
#
# OpenVLA convention used here:
#   state  = [q1..q7(padded), gripper]                       -> 8 dims
#   action = [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd] -> 7 dims
#
# For this 4-axis robot:
#   - state still uses padded joint state for observation
#   - action uses EE delta from meta.json["ee_pose"]
#   - ee_pose is assumed to be [x, y, z]
#   - rotational deltas are zero-filled (raw data has no EE orientation)
#   - gripper action uses raw step["action"][3] (0=open, 1=close)
#   - final step action uses zero delta + current gripper_cmd


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_float_list(values: List[Any]) -> List[float]:
    return [float(v) for v in values]


def pad_joint_state(joint_angles: List[Any], gripper_state: float, joint_pad_dim: int = 7) -> List[float]:
    joints = to_float_list(joint_angles)
    if len(joints) > joint_pad_dim:
        raise ValueError(f"joint_angles length {len(joints)} exceeds joint_pad_dim {joint_pad_dim}")
    padded = joints + [0.0] * (joint_pad_dim - len(joints))
    return padded + [float(gripper_state)]


def gripper_action_from_raw(step: Dict[str, Any]) -> float:
    """
    Raw step["action"] is [target_x, target_y, target_z, gripper_cmd].
    Keep the last element as gripper command.
    0.0=open, 1.0=close.
    """
    raw_action = step.get("action", [0.0, 0.0, 0.0, 0.0])
    if len(raw_action) < 4:
        return 0.0
    return 1.0 if float(raw_action[3]) >= 0.5 else 0.0


def ee_delta_action(
    curr_ee_pose: List[Any],
    next_ee_pose: Optional[List[Any]],
) -> List[float]:
    """
    Build EEF_POS action:
      [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
    Raw ee_pose is assumed to contain at least [x, y, z].
    Since raw data does not include EE orientation, rotational deltas are zero-filled.
    """
    curr = to_float_list(curr_ee_pose)
    if len(curr) < 3:
        raise ValueError(f"ee_pose must have at least 3 dims, got {len(curr)}")

    if next_ee_pose is None:
        dpos = [0.0, 0.0, 0.0]
    else:
        nxt = to_float_list(next_ee_pose)
        if len(nxt) < 3:
            raise ValueError(f"next ee_pose must have at least 3 dims, got {len(nxt)}")
        dpos = [float(n - c) for c, n in zip(curr[:3], nxt[:3])]

    return [dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0]


def is_idle_transition(
    curr_step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]],
    min_joint_delta_norm: float,
    min_gripper_delta: float,
    min_ee_delta_norm: float,
) -> bool:
    if next_step is None:
        return False

    curr_joint = to_float_list(curr_step.get("joint_angles", []))
    next_joint = to_float_list(next_step.get("joint_angles", []))
    if len(curr_joint) != len(next_joint):
        return False

    dq = [n - c for c, n in zip(curr_joint, next_joint)]
    joint_delta_norm = sum(v * v for v in dq) ** 0.5
    grip_delta = abs(float(next_step.get("gripper_state", 0.0)) - float(curr_step.get("gripper_state", 0.0)))

    curr_ee = to_float_list(curr_step.get("ee_pose", []))
    next_ee = to_float_list(next_step.get("ee_pose", []))
    if len(curr_ee) >= 3 and len(next_ee) >= 3:
        dee = [n - c for c, n in zip(curr_ee[:3], next_ee[:3])]
        ee_delta_norm = sum(v * v for v in dee) ** 0.5
    else:
        ee_delta_norm = float("inf")

    return (
        joint_delta_norm < min_joint_delta_norm
        and grip_delta < min_gripper_delta
        and ee_delta_norm < min_ee_delta_norm
    )


def copy_episode_images(
    raw_episode_dir: Path,
    out_images_dir: Path,
    referenced_files: List[str],
    image_mode: str = "copy",
) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if image_mode not in {"copy", "hardlink", "symlink"}:
        raise ValueError(f"Unsupported image_mode: {image_mode}")
    for image_file in referenced_files:
        src = raw_episode_dir / image_file
        dst = out_images_dir / image_file
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if image_mode == "hardlink":
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        elif image_mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)


def convert_episode(
    raw_episode_dir: Path,
    out_episode_dir: Path,
    joint_pad_dim: int = 7,
    include_failed: bool = False,
    drop_idle_steps: bool = False,
    min_joint_delta_norm: float = 1e-4,
    min_gripper_delta: float = 1e-4,
    min_ee_delta_norm: float = 1e-6,
    keep_debug_fields: bool = True,
    image_mode: str = "copy",
) -> Optional[Dict[str, Any]]:
    meta_path = raw_episode_dir / "meta.json"
    if not meta_path.exists():
        print(f"[WARN] skip {raw_episode_dir.name}: meta.json not found")
        return None

    meta = read_json(meta_path)
    success = bool(meta.get("success", False))
    if not include_failed and not success:
        print(f"[SKIP] {raw_episode_dir.name}: failed episode")
        return None

    raw_steps = meta.get("steps", [])
    if not raw_steps:
        print(f"[WARN] skip {raw_episode_dir.name}: empty steps")
        return None

    kept_indices: List[int] = []
    for i in range(len(raw_steps)):
        curr_step = raw_steps[i]
        next_step = raw_steps[i + 1] if i + 1 < len(raw_steps) else None

        if drop_idle_steps and i < len(raw_steps) - 1:
            if is_idle_transition(
                curr_step,
                next_step,
                min_joint_delta_norm,
                min_gripper_delta,
                min_ee_delta_norm,
            ):
                continue
        kept_indices.append(i)

    if not kept_indices:
        print(f"[WARN] skip {raw_episode_dir.name}: all steps filtered out")
        return None

    kept_steps = [raw_steps[i] for i in kept_indices]

    if kept_indices[-1] != len(raw_steps) - 1:
        kept_steps.append(raw_steps[-1])
        kept_indices.append(len(raw_steps) - 1)

    out_images_dir = out_episode_dir / "images"
    referenced_files = [str(step["image_file"]) for step in kept_steps]
    copy_episode_images(
        raw_episode_dir,
        out_images_dir,
        referenced_files,
        image_mode=image_mode,
    )

    episode_steps: List[Dict[str, Any]] = []
    num_steps = len(kept_steps)
    instruction = str(meta.get("instruction", ""))

    for local_i, raw_i in enumerate(kept_indices):
        curr = raw_steps[raw_i]
        # The action label must retain the collector's original control period.
        # When idle observations are filtered, using the next *kept* index here
        # merges several raw transitions into one large delta. That changes the
        # action semantics and produces labels that do not match the client's
        # per-step delta execution. Always supervise a retained observation with
        # its immediate next raw frame instead.
        next_raw_i = raw_i + 1 if raw_i + 1 < len(raw_steps) else None
        nxt = raw_steps[next_raw_i] if next_raw_i is not None else None

        state = pad_joint_state(
            joint_angles=curr.get("joint_angles", []),
            gripper_state=float(curr.get("gripper_state", 0.0)),
            joint_pad_dim=joint_pad_dim,
        )

        ee_delta = ee_delta_action(
            curr_ee_pose=curr.get("ee_pose", []),
            next_ee_pose=(nxt.get("ee_pose", []) if nxt is not None else None),
        )
        grip_cmd = gripper_action_from_raw(curr)
        action = ee_delta + [grip_cmd]

        is_last = local_i == (num_steps - 1)
        is_terminal = bool(is_last and success)
        reward = 1.0 if is_terminal else 0.0
        discount = 0.0 if is_terminal else 1.0

        step_item: Dict[str, Any] = {
            "observation": {
                "image": f"images/{curr['image_file']}",
                "state": state,
            },
            "action": action,
            "language_instruction": instruction,
            "reward": reward,
            "discount": discount,
            "is_first": local_i == 0,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "timestep": int(curr.get("t", local_i)),
            "raw_index": int(raw_i),
            "action_target_raw_index": int(next_raw_i) if next_raw_i is not None else None,
            "raw_waypoint_action": to_float_list(curr.get("action", [])),
        }

        if keep_debug_fields:
            step_item["observation"]["joint_angles_raw"] = to_float_list(curr.get("joint_angles", []))
            step_item["observation"]["gripper_state_raw"] = float(curr.get("gripper_state", 0.0))
            step_item["observation"]["ee_pose"] = to_float_list(curr.get("ee_pose", []))
            step_item["observation"]["object_pose"] = to_float_list(curr.get("object_pose", []))

        episode_steps.append(step_item)

    episode_json = {
        "episode_metadata": {
            "episode_id": int(meta.get("episode_id", -1)),
            "instruction": instruction,
            "success": success,
            "goal_xy": to_float_list(meta.get("goal_xy", [])),
            "box_init_xy": to_float_list(meta.get("box_init_xy", [])),
            "box_init_yaw": float(meta.get("box_init_yaw", 0.0)),
            "task_type": str(meta.get("task_type", "")),
            "target_color": meta.get("target_color"),
            "target_object_type": meta.get("target_object_type"),
            "target_body_name": meta.get("target_body_name"),
            "dataset_generator_version": meta.get("dataset_generator_version"),
            "collection_config": meta.get("collection_config"),
            "raw_episode_dir": raw_episode_dir.name,
            "num_steps_raw": len(raw_steps),
            "num_steps_converted": len(episode_steps),
            "joint_state_dim": joint_pad_dim + 1,
            "eef_action_dim": 7,
            "action_semantics": {
                "type": "EEF_POS",
                "ee_position_action": (
                    "immediate_raw_next_ee_pose[:3] - current_raw_ee_pose[:3]; "
                    "idle filtering never merges multiple raw transitions"
                ),
                "ee_rotation_action": "[0,0,0] because raw data has no EE orientation deltas",
                "gripper_action": "raw step['action'][3] mapped to 0=open, 1=close",
            },
        },
        "steps": episode_steps,
    }

    write_json(out_episode_dir / "episode.json", episode_json)

    return {
        "episode_id": int(meta.get("episode_id", -1)),
        "instruction": instruction,
        "success": success,
        "raw_episode_dir": raw_episode_dir.name,
        "relative_episode_json": str((out_episode_dir / "episode.json").as_posix()),
        "num_steps_raw": len(raw_steps),
        "num_steps_converted": len(episode_steps),
    }


def make_split_lists(episode_dirs: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    episode_dirs = list(episode_dirs)
    rng = random.Random(seed)
    rng.shuffle(episode_dirs)

    if val_ratio <= 0.0:
        return sorted(episode_dirs), []

    val_count = int(round(len(episode_dirs) * val_ratio))
    val_count = max(1, val_count) if len(episode_dirs) > 1 else 0
    val_dirs = sorted(episode_dirs[:val_count])
    train_dirs = sorted(episode_dirs[val_count:])
    return train_dirs, val_dirs


def convert_dataset(
    raw_root: Path,
    out_root: Path,
    joint_pad_dim: int = 7,
    include_failed: bool = False,
    val_ratio: float = 0.0,
    seed: int = 42,
    drop_idle_steps: bool = False,
    min_joint_delta_norm: float = 1e-4,
    min_gripper_delta: float = 1e-4,
    min_ee_delta_norm: float = 1e-6,
    keep_debug_fields: bool = True,
    task_type: Optional[str] = None,
    image_mode: str = "copy",
) -> None:
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted([p for p in raw_root.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found under: {raw_root}")
    if task_type is not None:
        filtered_dirs = []
        for episode_dir in episode_dirs:
            meta_path = episode_dir / "meta.json"
            if not meta_path.exists():
                continue
            if str(read_json(meta_path).get("task_type", "")) == task_type:
                filtered_dirs.append(episode_dir)
        episode_dirs = filtered_dirs
        if not episode_dirs:
            raise FileNotFoundError(
                f"No task_type={task_type!r} episodes found under: {raw_root}"
            )

    train_dirs, val_dirs = make_split_lists(episode_dirs, val_ratio=val_ratio, seed=seed)

    train_manifest: List[Dict[str, Any]] = []
    val_manifest: List[Dict[str, Any]] = []

    split_map = [("train", train_dirs, train_manifest)]
    if val_dirs:
        split_map.append(("val", val_dirs, val_manifest))

    for split_name, split_dirs, manifest in split_map:
        for raw_episode_dir in split_dirs:
            out_episode_dir = out_root / split_name / raw_episode_dir.name
            out_episode_dir.mkdir(parents=True, exist_ok=True)

            result = convert_episode(
                raw_episode_dir=raw_episode_dir,
                out_episode_dir=out_episode_dir,
                joint_pad_dim=joint_pad_dim,
                include_failed=include_failed,
                drop_idle_steps=drop_idle_steps,
                min_joint_delta_norm=min_joint_delta_norm,
                min_gripper_delta=min_gripper_delta,
                min_ee_delta_norm=min_ee_delta_norm,
                keep_debug_fields=keep_debug_fields,
                image_mode=image_mode,
            )
            if result is None:
                shutil.rmtree(out_episode_dir, ignore_errors=True)
                continue

            result["split"] = split_name
            result["relative_episode_json"] = str((Path(split_name) / raw_episode_dir.name / "episode.json").as_posix())
            manifest.append(result)
            print(
                f"[OK] {split_name}/{raw_episode_dir.name} | "
                f"steps {result['num_steps_raw']} -> {result['num_steps_converted']} | success={result['success']}"
            )

    write_jsonl(out_root / "manifest_train.jsonl", train_manifest)
    if val_dirs:
        write_jsonl(out_root / "manifest_val.jsonl", val_manifest)

    dataset_info = {
        "format_name": "openvla_rlds_intermediate_eef_pos",
        "raw_root": str(raw_root),
        "joint_state_dim": joint_pad_dim + 1,
        "eef_action_dim": 7,
        "joint_dims_padded_to": joint_pad_dim,
        "train_episodes": len(train_manifest),
        "val_episodes": len(val_manifest),
        "include_failed": include_failed,
        "drop_idle_steps": drop_idle_steps,
        "min_joint_delta_norm": min_joint_delta_norm,
        "min_gripper_delta": min_gripper_delta,
        "min_ee_delta_norm": min_ee_delta_norm,
        "task_type_filter": task_type,
        "image_mode": image_mode,
        "schema": {
            "episode_json": {
                "episode_metadata": {
                    "episode_id": "int",
                    "instruction": "str",
                    "success": "bool",
                    "goal_xy": "list[float]",
                    "box_init_xy": "list[float]",
                    "box_init_yaw": "float",
                    "raw_episode_dir": "str",
                    "num_steps_raw": "int",
                    "num_steps_converted": "int",
                },
                "steps": {
                    "observation.image": "relative image path",
                    "observation.state": f"list[float] length {joint_pad_dim + 1}",
                    "action": "list[float] length 7 = [dx,dy,dz,droll,dpitch,dyaw,gripper_cmd]",
                    "language_instruction": "str",
                    "reward": "float",
                    "discount": "float",
                    "is_first": "bool",
                    "is_last": "bool",
                    "is_terminal": "bool",
                    "timestep": "int",
                },
            }
        },
        "notes": [
            "state = [joint_angles padded to 7, gripper_state]",
            (
                "action = [immediate raw next_ee_pose[:3] - current raw ee_pose[:3], "
                "0,0,0, gripper_cmd]; idle filtering never merges transitions"
            ),
            "rotational deltas are zero-filled because raw data does not include EE orientation deltas",
            "raw waypoint action is preserved in each step as raw_waypoint_action for debugging",
            "This is an intermediate format; your TFDS/RLDS builder should load observation.image from disk and emit actual RLDS records.",
        ],
    }
    write_json(out_root / "dataset_info.json", dataset_info)

    print("\nDone.")
    print(f"train episodes: {len(train_manifest)}")
    print(f"val episodes  : {len(val_manifest)}")
    print(f"output root   : {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw robot episodes to OpenVLA RLDS intermediate format.")
    parser.add_argument("--raw_root", type=str, default="./dataset_raw_4color_dynamic_center_camera_visible_grasp", help="Root directory containing episode_*/meta.json")
    parser.add_argument("--out_root", type=str, default="./rlds_out", help="Output directory for intermediate dataset")
    parser.add_argument("--joint_pad_dim", type=int, default=7, help="Pad joint states to this many joints")
    parser.add_argument("--include_failed", action="store_true", help="Include failed episodes too")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation split ratio, e.g. 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--drop_idle_steps", action="store_true", help="Drop tiny-motion transitions")
    parser.add_argument("--min_joint_delta_norm", type=float, default=1e-4, help="Idle threshold for joint delta norm")
    parser.add_argument("--min_gripper_delta", type=float, default=1e-4, help="Idle threshold for gripper delta")
    parser.add_argument("--min_ee_delta_norm", type=float, default=1e-6, help="Idle threshold for ee delta norm")
    parser.add_argument("--no_debug_fields", action="store_true", help="Do not keep ee/object/raw debug fields")
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Only convert episodes whose meta.json task_type matches this value",
    )
    parser.add_argument(
        "--image_mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How intermediate image files reference raw PNGs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        joint_pad_dim=args.joint_pad_dim,
        include_failed=args.include_failed,
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_idle_steps=args.drop_idle_steps,
        min_joint_delta_norm=args.min_joint_delta_norm,
        min_gripper_delta=args.min_gripper_delta,
        min_ee_delta_norm=args.min_ee_delta_norm,
        keep_debug_fields=not args.no_debug_fields,
        task_type=args.task_type,
        image_mode=args.image_mode,
    )
