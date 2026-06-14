from pathlib import Path

from convert_raw_to_openvla_rlds_intermediate import convert_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = SCRIPT_DIR.parent
OUTPUT_ROOT = SCRIPT_DIR / "task_balanced_intermediate"


DATASETS = {
    "raccoon_grasp": {
        "raw_root": MUJOCO_DIR / "raccoon_grasp_colored_objects",
        "task_type": "grasp",
    },
    "raccoon_push": {
        "raw_root": MUJOCO_DIR / "raccoon_multitask_colored_objects",
        "task_type": "push",
    },
    "raccoon_pick_and_place": {
        "raw_root": MUJOCO_DIR / "raccoon_multitask_colored_objects",
        "task_type": "pick_and_place",
    },
}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name, config in DATASETS.items():
        out_root = OUTPUT_ROOT / dataset_name
        print(
            f"\n=== {dataset_name}: raw={config['raw_root']} "
            f"task={config['task_type']} ==="
        )
        convert_dataset(
            raw_root=config["raw_root"],
            out_root=out_root,
            val_ratio=0.1,
            seed=42,
            drop_idle_steps=True,
            keep_debug_fields=False,
            task_type=config["task_type"],
            image_mode="hardlink",
        )


if __name__ == "__main__":
    main()
