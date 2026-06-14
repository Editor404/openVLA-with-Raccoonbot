from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow_datasets as tfds


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = SCRIPT_DIR.parent
REPO_DIR = MUJOCO_DIR.parent
INTERMEDIATE_BASE = MUJOCO_DIR / "raccoon_dataset" / "task_balanced_intermediate"
DEFAULT_DATA_DIR = REPO_DIR / "tensorflow_datasets"

sys.path.insert(0, str(SCRIPT_DIR / "raccoon_pick_place"))
from raccoon_pick_place_dataset_builder import RaccoonPickPlace  # noqa: E402


class RaccoonGrasp(RaccoonPickPlace):
    INTERMEDIATE_ROOT = INTERMEDIATE_BASE / "raccoon_grasp"
    DATASET_DESCRIPTION = "RaccoonBot colored-object grasp demonstrations."


class RaccoonPush(RaccoonPickPlace):
    INTERMEDIATE_ROOT = INTERMEDIATE_BASE / "raccoon_push"
    DATASET_DESCRIPTION = "RaccoonBot colored-object push demonstrations."


class RaccoonPickAndPlace(RaccoonPickPlace):
    INTERMEDIATE_ROOT = INTERMEDIATE_BASE / "raccoon_pick_and_place"
    DATASET_DESCRIPTION = "RaccoonBot colored-object pick-and-place demonstrations."


BUILDERS = {
    "raccoon_grasp": RaccoonGrasp,
    "raccoon_push": RaccoonPush,
    "raccoon_pick_and_place": RaccoonPickAndPlace,
}


def count_manifest_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_intermediate(dataset_name: str, builder_cls) -> tuple[int, int]:
    root = Path(builder_cls.INTERMEDIATE_ROOT)
    train_count = count_manifest_lines(root / "manifest_train.jsonl")
    val_count = count_manifest_lines(root / "manifest_val.jsonl")
    if train_count == 0:
        raise FileNotFoundError(
            f"{dataset_name}: train manifest가 없거나 비어 있습니다: {root}"
        )
    return train_count, val_count


def build_dataset(dataset_name: str, builder_cls, data_dir: Path, overwrite: bool):
    train_count, val_count = validate_intermediate(dataset_name, builder_cls)
    print(
        f"\n=== Build {dataset_name} ===\n"
        f"intermediate={builder_cls.INTERMEDIATE_ROOT}\n"
        f"expected train={train_count}, val={val_count}\n"
        f"tfds data_dir={data_dir}"
    )

    builder = builder_cls(data_dir=str(data_dir))
    download_config = tfds.download.DownloadConfig(
        download_mode=(
            tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
            if not overwrite
            else tfds.GenerateMode.FORCE_REDOWNLOAD
        )
    )
    builder.download_and_prepare(download_config=download_config)

    built_train = int(builder.info.splits["train"].num_examples)
    built_val = (
        int(builder.info.splits["val"].num_examples)
        if "val" in builder.info.splits
        else 0
    )
    if (built_train, built_val) != (train_count, val_count):
        raise RuntimeError(
            f"{dataset_name}: TFDS count mismatch: "
            f"built=({built_train}, {built_val}), "
            f"expected=({train_count}, {val_count})"
        )
    print(
        f"[OK] {dataset_name}: train={built_train}, val={built_val}, "
        f"path={builder.data_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build separate grasp/push/pick-and-place TFDS datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="TFDS output root",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(BUILDERS),
        default=list(BUILDERS),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild datasets even if an existing version is present",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        build_dataset(
            dataset_name,
            BUILDERS[dataset_name],
            args.data_dir.resolve(),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
