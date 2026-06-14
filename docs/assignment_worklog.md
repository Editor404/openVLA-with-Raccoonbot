# Assignment Work Log

## Goal

Extend the RaccoonBot OpenVLA pipeline for the FAIR Lab assignment by adding dataset diversity, improving code clarity/reliability, and collecting reproducible evidence without committing large datasets or checkpoints.

## Submission Constraints

- Deadline: 2026-06-07 23:59.
- Include modified source, README, logs, screenshots/episode visualizations, and a short report.
- Do not upload large generated datasets or model checkpoints.

## Planned Scope

### Dataset extension

- Add non-cylinder object demonstrations, starting with cube and sphere.
- Add diverse instruction templates beyond `grasp the {color} cylinder`.
- Regenerate a small MuJoCo demonstration set.
- Rebuild RLDS / TFDS and visualize one episode.
- Run a short LoRA sanity test.

### Code improvement

- Improve 7D OpenVLA action to 4DOF RaccoonBot execution mapping.
- Add timing/action logs for clearer before/after evidence.

## Daily Notes

### 2026-06-03

- Re-explored the local directory structure.
- Confirmed `Raccoonbot_Openvla/` is the Git repository and `checkpoint/` is a local 15GB model artifact outside the repository.
- Created submission artifact folders under `results/` and this work log under `docs/`.
- Added a baseline server launch script to make inference evidence collection reproducible.

Validation evidence collected today:

- `results/logs/checkpoint_metadata_baseline.txt` confirms the local checkpoint has OpenVLA action-prediction metadata and `dataset_statistics.json` contains `raccoon_pick_place`.
- `results/logs/baseline_server_dry_run.txt` confirms the baseline server command resolves the repository, checkpoint, and output log paths without launching the long-running model server.

Notebook-grounded update:

- The actual executed notebook is `/home/keivn/Downloads/raccoon_lab1.ipynb`.
- The Jupyter server project path used in early cells is `/data/yb/Raccoonbot_Openvla`.
- Future runs must normalize all notebook commands to `/data/yb/Raccoonbot_Openvla`; do not use `/data/Raccoonbot_Openvla`.
- The notebook TFDS build produced 360 train examples and 40 val examples for `raccoon_pick_place`.
- The notebook short LoRA run completed step 100 and saved a checkpoint under `/data/yb/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-eef-v100--image_aug`.
- The notebook also downloaded `fair-lab/openvla-7b-finetuned-raccoonbot` and served it with `--default-unnorm-key raccoon_pick_place`.

Dataset extension implementation update:

- Updated `Mujoco/raccoon_grasp_multicolor_scene_dataset.py` to support target object types `cylinder`, `cube`, and `sphere` without requiring a new XML file.
- Added an instruction template pool for diverse language instructions.
- Added `target_object_type` to raw `meta.json` episode metadata and `object_type` to `all_object_init_poses` metadata.
- Changed dataset balancing from color-only to `(color, object_type)` target pairs.
- Smoke-tested object spec sampling and MuJoCo reset with a forced red cube target.

### 2026-06-14

#### Grasp dataset generation

- Extended the grasp dataset to four colors (`red`, `blue`, `green`, `yellow`)
  and three object types (`cylinder`, `cube`, `sphere`).
- Configured a balanced target of 1,200 successful episodes:
  100 episodes for each of the 12 color/object combinations.
- Resolved paths relative to the Python script so local execution writes into
  the repository's MuJoCo directory rather than a server-specific path.
- Expanded the object placement workspace in the Y direction to reduce random
  placement failures.
- Changed placement failure handling from terminating the entire collection
  process to rejecting the sample and retrying.
- Corrected grasp execution so the gripper remains open while approaching the
  object and closes only after the end effector reaches the grasp-height
  alignment gate.
- Disabled artificial grasp stabilization by default. Successful grasps now
  rely on MuJoCo contacts and physics rather than directly moving the object's
  pose with the gripper.
- Added regression tests for action safety and open-gripper approach behavior.
- At the latest deadline discussion, 1,124 grasp episodes had been generated.
  This is 93.7% of the planned 1,200 episodes and is sufficient for a
  deadline-constrained training run if necessary.

#### Placement and trajectory failures observed

- Original object placement occasionally failed with:
  `색상 object들을 겹치지 않게 배치하지 못했습니다`.
- Root cause: four objects had to satisfy minimum spacing and target-clearance
  constraints inside a narrow random sampling region.
- Individual attempts also occasionally exceeded the recorded end-effector
  transition limit of 0.005 m. These attempts were rejected rather than stored
  as valid demonstrations.
- GPU utilization remained at 0% during MuJoCo dataset collection even though
  approximately 30 GB of GPU memory was reserved. The collection workload is
  dominated by CPU simulation, rendering, PNG encoding, and filesystem I/O.
- Local collection was therefore faster than collection on the remote system.

#### Multitask dataset extension

- Created:
  `Mujoco/raccoon_multitask_colored_objects_dataset.py`.
- Added two task types required for broader task diversity:
  - `push`
  - `pick_and_place`
- Configured 720 total successful episodes:
  - push: 360
  - pick-and-place: 360
  - 30 episodes for each of the 24 task/color/object combinations
- Added task-specific language instruction templates.
- Added task-specific Cartesian plans and interpolated IK prechecks.
- For push, the gripper closes before contact and pushes forward at a
  0.030 m commanded height.
- Push targets are sampled directly inside a reliable center corridor instead
  of depending on repeated rejection from the full X range.
- If no valid target can be sampled within one attempt, collection skips the
  attempt and continues rather than terminating the whole process.
- Added resumable collection:
  - Existing successful `meta.json` files are scanned.
  - Per-task/color/object success counts are restored.
  - New episodes start after the largest existing episode ID.
  - Existing episode directories are not overwritten.
- Resume was verified when 51 episodes existed:
  `loaded_successes=51`, `next_episode_id=52`, `skipped=0`.
- Later progress showed approximately 261 successful multitask episodes.
- The sampler should prioritize combinations with the smallest counts when
  collecting a deadline-limited balanced subset.

#### Validation evidence

- `raccoon_multitask_colored_objects_dataset.py` passes Python compilation.
- `test_multitask_colored_objects.py` contains six regression tests covering:
  - balanced target counts
  - push gripper sequence
  - pick-and-place gripper sequence
  - IK-valid center-workspace plans
  - forced center-corridor push sampling
  - resume count and episode-ID recovery
- Latest targeted result: `Ran 6 tests ... OK`.

#### Dataset and training strategy

- Available datasets are intentionally unequal in raw size:
  - grasp: planned 1,200
  - push: planned 360
  - pick-and-place: planned 360
- A sequential second-stage fine-tune using only push and pick-and-place data
  may cause catastrophic forgetting of grasp behavior.
- Recommended training strategy:
  1. Train or fine-tune on the grasp dataset.
  2. Continue fine-tuning using replayed grasp samples mixed with push and
     pick-and-place samples.
  3. Use task-balanced sampling so each task contributes approximately one
     third of training batches despite unequal raw dataset sizes.
  4. Use a lower learning rate for the continuation stage.
  5. Evaluate grasp, push, and pick-and-place separately.
- Under the submission deadline, completing training and showing inference is
  more important than reaching every planned raw episode count.

Task-balanced conversion implementation:

- Converted the completed raw data into three separate intermediate datasets:
  - `raccoon_grasp`: 1,080 train / 120 validation
  - `raccoon_push`: 146 train / 16 validation
  - `raccoon_pick_and_place`: 131 train / 15 validation
- Intermediate images use hard links to avoid physically duplicating the raw
  PNG data.
- Added `Mujoco/rlds_dataset_builder/build_task_balanced_tfds.py` to build all
  three TFDS datasets into `Raccoonbot_Openvla/tensorflow_datasets/`.
- Registered the three datasets in OpenVLA and added the
  `raccoon_task_balanced` mixture with equal weights `1.0 : 1.0 : 1.0`.
- Added a standardization transform that converts collector gripper labels
  (`0=open`, `1=close`) into the OpenVLA convention (`1=open`, `0=close`).
- Local static and registration tests passed. The final TFDS materialization
  must be run in the TensorFlow/RLDS environment because the local collection
  environment does not have TensorFlow installed.

#### Time estimates discussed

- At roughly three minutes per successful episode, generating hundreds of
  remaining episodes requires approximately one day.
- With 261 of 720 multitask episodes completed, 459 remained:
  approximately 22 hours 57 minutes without retries, and potentially around
  30 hours given the observed success rate.
- The recommendation under deadline pressure was to stop collection and use
  the already generated grasp and multitask demonstrations for training,
  validation, and inference evidence.

#### Required submission logs

Lecture 19 states that the repository should contain modified code, README,
result screenshots or episode visualizations, training/inference logs, and a
short report. Large datasets and model checkpoints must not be committed.

Recommended small artifacts:

```text
results/logs/
├── tfds_build.log
├── training.log
├── inference_server.log
└── inference_client.log
```

- `training.log` should show the dataset name, command/configuration, training
  steps or epochs, loss, learning rate, checkpoint path, and completion/errors.
- `inference_server.log` should show the loaded checkpoint and server startup.
- `inference_client.log` should show instructions, predicted actions, episode
  outcomes, timing/step counts, and errors.
- Capture reproducible terminal output with `2>&1 | tee <log path>`.

#### Report-ready contribution summary

1. Added object diversity across four colors and three shapes.
2. Added language diversity with multiple instruction templates.
3. Added push and pick-and-place task extensions.
4. Fixed physically invalid closed-gripper approach demonstrations.
5. Removed default artificial object attachment and required physical contact.
6. Improved robustness through placement retries, IK prechecks, safety limits,
   and resumable balanced collection.
7. Added regression tests and defined reproducible training/inference evidence
   requirements.
