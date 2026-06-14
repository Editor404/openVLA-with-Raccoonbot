# Task-balanced TFDS build

## 1. Intermediate conversion

```bash
cd Raccoonbot_Openvla/Mujoco/raccoon_dataset
python3 prepare_task_balanced_intermediate.py \
  2>&1 | tee ../../results/logs/intermediate_conversion.log
```

Expected episode counts:

- `raccoon_grasp`: train 1080, val 120
- `raccoon_push`: train 146, val 16
- `raccoon_pick_and_place`: train 131, val 15

## 2. TFDS build

Run this inside the TensorFlow/RLDS environment:

```bash
cd Raccoonbot_Openvla/Mujoco/rlds_dataset_builder
python3 build_task_balanced_tfds.py --overwrite \
  2>&1 | tee ../../results/logs/tfds_build.log
```

Output:

```text
Raccoonbot_Openvla/tensorflow_datasets/
├── raccoon_grasp/
├── raccoon_push/
└── raccoon_pick_and_place/
```

## 3. Fine-tuning mixture

Use:

```text
--dataset_name raccoon_task_balanced
```

The registered mixture weights are `1.0 : 1.0 : 1.0`, so grasp, push, and
pick-and-place are sampled with equal task probability without duplicating the
smaller raw datasets.
