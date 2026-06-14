# Raccoonbot_Openvla

## Assignment Submission Summary

This repository extends the FAIR Lab RaccoonBot OpenVLA pipeline with:

- four colors and three object shapes (`cylinder`, `cube`, `sphere`);
- grasp, push, and pick-and-place demonstrations;
- multiple task-specific language instruction templates;
- task-balanced OpenVLA dataset registration;
- corrected one-frame action labels and OpenVLA gripper conventions;
- safer physical demonstrations, IK checks, retry handling, and resumable collection.

Submission evidence is stored under `results/`:

- `results/episode_visualizations/raccoon_grasp_episode_grid.png`
- `results/logs/tfds_build.log`
- `results/logs/training.log`
- `results/logs/inference_server.log`
- `report.pdf`

Generated TFDS datasets, intermediate images, model weights, and checkpoints are
intentionally excluded from Git.

## FAIR Lab Assignment Progress

This repository is being extended for the FAIR Lab OpenVLA assignment.

Current submission scope:

- Dataset extension: add object diversity beyond cylinders and add more instruction templates.
- Code improvement: improve 7D OpenVLA action to 4DOF RaccoonBot execution mapping and add timing/action logs.
- Evidence: include small logs, screenshots, episode visualizations, and a short report.
- Large files: do not commit generated datasets, TFRecords, OpenVLA checkpoints, or `*.safetensors`.

Workspace note:

- The local fine-tuned model checkpoint is expected outside this Git repository at `../checkpoint`.
- Use `scripts/run_checkpoint_server.sh` to launch the baseline inference server against that checkpoint.
- Small submission artifacts belong under `results/`.
- Daily implementation notes are tracked in `docs/assignment_worklog.md`.

Baseline server command:

```bash
CHECKPOINT_DIR=/home/keivn/openVLA/checkpoint ./scripts/run_checkpoint_server.sh
```

Dry-run command for path validation:

```bash
DRY_RUN=1 CHECKPOINT_DIR=/home/keivn/openVLA/checkpoint ./scripts/run_checkpoint_server.sh
```


## Assignment Execution Plan / Resume Notes

Use this section as the persistent handoff for future sessions. The assignment deadline is **2026-06-07 23:59**. The repository must include modified code, README, logs, screenshots or episode visualizations, and a short report. Do **not** commit generated datasets, TFRecords, model checkpoints, or `*.safetensors`.

### Current State as of 2026-06-03

- Git repository: `Raccoonbot_Openvla/`
- Local workstation root: `/home/keivn/openVLA`
- External Jupyter Lab root from actual notebook: `/data/yb/Raccoonbot_Openvla`
- Local checkpoint: `../checkpoint`
- Checkpoint size: about 15GB, intentionally outside the Git repository
- Baseline checkpoint metadata saved at `results/logs/checkpoint_metadata_baseline.txt`
- Baseline server dry-run saved at `results/logs/baseline_server_dry_run.txt`
- Baseline launch helper: `scripts/run_checkpoint_server.sh`
- Work log: `docs/assignment_worklog.md`
- Submission artifact folders:
  - `results/logs/`
  - `results/screenshots/`
  - `results/episode_visualizations/`

### Actual Jupyter Lab Run Path from `raccoon_lab1.ipynb`

The real notebook that was used is `/home/keivn/Downloads/raccoon_lab1.ipynb`. Its server-side workspace path is **`/data/yb`**. For all future runs, use only paths under `/data/yb`; do not rely on `/data/Raccoonbot_Openvla` compatibility paths. The project path is:

```text
/data/yb/Raccoonbot_Openvla/
```

Run from the project root on the Jupyter server:

```bash
cd /data/yb/Raccoonbot_Openvla
```

Baseline checkpoint server using the Hugging Face checkpoint path from the notebook:

```bash
CHECKPOINT_DIR=/data/yb/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot ./scripts/run_checkpoint_server.sh
```

If the checkpoint is copied beside this repo instead, use:

```bash
CHECKPOINT_DIR=/data/yb/checkpoint ./scripts/run_checkpoint_server.sh
```

Dry-run path validation:

```bash
DRY_RUN=1 CHECKPOINT_DIR=/data/yb/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot ./scripts/run_checkpoint_server.sh
```

Actual notebook evidence to preserve:

- TFDS build created `raccoon_pick_place` with 360 train examples and 40 val examples.
- Short LoRA run used `max_steps=100`, `save_steps=100`, `batch_size=8`, `grad_accumulation_steps=2`, `lora_rank=32`.
- LoRA output directory was `openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-eef-v100--image_aug`.
- Hugging Face checkpoint download used `fair-lab/openvla-7b-finetuned-raccoonbot`.
- Inference server used `--default-unnorm-key raccoon_pick_place`.

### Chosen Assignment Scope

#### Dataset Extension

Implement a conservative but meaningful extension:

1. Add new object types beyond cylinder:
   - `cube`
   - `sphere`
2. Keep grasp as the primary task for reliability.
3. Add diverse language instructions:
   - `grasp the {color} {object}`
   - `pick up the {color} {object}`
   - `grab the {color} {object}`
   - `hold the {color} {object}`
   - `move to the {color} {object} and grasp it`
4. Generate a small new MuJoCo demonstration set.
5. Rebuild RLDS / TFDS.
6. Visualize one episode.
7. Run a short LoRA sanity test.

#### Code Improvement

Implement at least these two improvements:

1. Improve 7D OpenVLA action to 4DOF RaccoonBot execution mapping.
   - Explicitly use `dx, dy, dz, gripper`.
   - Log ignored `droll, dpitch, dyaw` rotation components.
   - Add clipping / workspace bounds where appropriate.
2. Add timing/action logs.
   - Inference latency.
   - Raw predicted 7D action.
   - Clipped/mapped 4DOF command.
   - Gripper command.
   - Step index where applicable.

### Step-by-Step Work Plan

#### Phase 1 — Dataset object/language extension

Target files:

- `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`
- `Mujoco/Raccoon_colored_cylinder.xml` or a new XML variant if needed
- `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`
- `Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py`

Tasks:

- Add object type selection to dataset generation.
- Add cube/sphere object specs in MuJoCo scene generation or XML handling.
- Store `target_object_type` in `meta.json` and converted `episode.json` metadata.
- Replace the single fixed instruction template with an instruction template pool.
- Balance generated demonstrations by `(color, object_type)` pair where practical.

Expected evidence:

- `results/logs/demo_generation.log`
- `results/episode_visualizations/extended_episode_grid.png`
- README snippet showing example metadata with `target_color` and `target_object_type`

#### Phase 2 — RLDS / TFDS rebuild

Target files:

- `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`
- `Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py`

Tasks:

- Convert raw generated episodes into OpenVLA RLDS intermediate format.
- Remove or soften hard-coded `/data/yb/Raccoonbot_Openvla/...` path by supporting an environment variable, e.g. `RACCOON_INTERMEDIATE_ROOT`.
- Run `tfds build --overwrite`.

Expected evidence:

- `results/logs/rlds_conversion.log`
- `results/logs/tfds_build.log`
- `results/screenshots/tfds_dataset_info.png` if available

#### Phase 3 — Short LoRA sanity test

Target file:

- `openvla/vla-scripts/finetune.py` only if necessary; prefer no change unless needed.

Recommended command shape:

```bash
cd openvla
export PYTHONPATH=$(pwd):$PYTHONPATH
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/tensorflow_datasets \
  --dataset_name raccoon_pick_place \
  --run_root_dir ./openvla-runs \
  --adapter_tmp_dir ./openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --max_steps 5 \
  --save_steps 5 \
  --run_id_note assignment-short-test
```

Expected evidence:

- `results/logs/lora_short_test.log`
- Do not commit `openvla-runs/`, adapter outputs, or checkpoints.

#### Phase 4 — 7D-to-4DOF mapping improvement

Target file:

- `Mujoco/raccoon_env.py`

Tasks:

- Make the mapping from OpenVLA 7D action to RaccoonBot execution explicit.
- Keep execution focused on xyz delta and gripper.
- Add structured return/log data such as:

```text
raw_action_7d
used_delta_xyz
ignored_rotation_deltas
clipped_target_xyz
gripper_cmd
```

Expected evidence:

- Before/after code diff.
- `results/logs/action_mapping_before_after.log`
- Short discussion in report: why this makes the system clearer or more reliable.

#### Phase 5 — Inference timing/action logging

Target files:

- `openvla/openvla_server.py`
- optionally `Mujoco/raccoon_env.py`

Tasks:

- Add timing logs around preprocessing, model inference, and total request time.
- Keep logs lightweight and text-based for submission.
- Preserve existing `/health` and `/predict` API behavior.

Expected evidence:

- `results/logs/baseline_inference_server.txt`
- `results/logs/improved_inference_server.txt`
- Before/after latency/action table in README or report.

#### Phase 6 — README and report

Target files:

- `README.md`
- `docs/assignment_worklog.md`
- `docs/short_report.md`
- `report.pdf` before final submission

README should include:

- What changed.
- How to generate demonstrations.
- How to convert/rebuild RLDS/TFDS.
- How to run short LoRA test.
- How to run inference server.
- Where results are stored.
- Note that checkpoints/datasets are excluded.

Report should include:

- Problem and original limitation.
- Dataset extension design.
- Code improvement design.
- Before/after evidence.
- Effect on the VLA pipeline.
- Limitations and future work.

### Acceptance Criteria

- [ ] At least one new object type works in generated demonstrations.
- [ ] Multiple instruction templates are used and visible in metadata.
- [ ] New MuJoCo demonstrations are generated.
- [ ] RLDS / TFDS rebuild completes or a clear log explains any environment blocker.
- [ ] One episode visualization is saved.
- [ ] Short LoRA test runs or a clear environment blocker log is saved.
- [ ] At least one meaningful code improvement is implemented; target is two improvements.
- [ ] Before/after evidence is saved under `results/`.
- [ ] README explains changed files, run commands, and results.
- [ ] `report.pdf` exists before final submission.
- [ ] No large datasets/checkpoints/safetensors are committed.

### Next Immediate Step

Start with Phase 1:

1. Edit `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`.
2. Add object type and instruction template support.
3. Generate a very small smoke-test demo set first, e.g. 4 to 12 successful episodes.
4. Save generation output to `results/logs/demo_generation.log`.

⭐ 1~3번은 직접 finetuning을 진행하는 내용이니 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행<br>

0~3번 server에서 실행, 4번 local-server 실행<br>


## 0. Dependencies
```
git clone https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla.git
```

필요한 패키지 설치
```
apt update
apt install -y \
  libegl1 \
  libgl1 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libgles2 \
  libegl1-mesa \
  libegl1-mesa-dev \
  mesa-utils

cd Raccoonbot_Openvla/openvla
pip install .
```

## 1. Dataset 생성
MuJoCo 가상환경에서 finetuning을 위한 데이터를 수집 <br>
(main 함수 `num_episodes`으로 dataset sample 수 변경 가능)
```
cd /data/yb/Raccoonbot_Openvla/Mujoco
python raccoon_grasp_multicolor_scene_dataset.py
```
실행하면 /data/yb/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder 하위에 episode별로 dataset png 확인 가능

## 2. rlds 파일 변환
raw data를 rlds builder에 맞게 변경
아래 명령문 그대로 실행
```
cd /data/yb/Raccoonbot_Openvla/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
--raw_root /data/yb/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder \
--out_root /data/yb/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
--val_ratio 0.1
```

## 2-1. rlds builder
rlds builder 실행
아래 명령문 그대로 실행
```
cd /data/yb/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```
실행하면 root 하위에 tensorflow_datasets 폴더 생성됨
```
mv /root/tensorflow_datasets /data/yb/Raccoonbot_Openvla/
```

## 3. Raccoonbot 기반 OpenVLA finetuning
아래 명령어 그대로 실행 <br>
(`max_steps`, `save_steps` 변경 가능)
```
cd /data/yb/Raccoonbot_Openvla/openvla
export PYTHONPATH=/data/yb/Raccoonbot_Openvla/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/yb/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_pick_place \
  --run_root_dir /data/yb/Raccoonbot_Openvla/openvla/openvla-runs \
  --adapter_tmp_dir /data/yb/Raccoonbot_Openvla/openvla/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_id_note raccoon-eef-v100
```

## 4. Mujoco 환경 Inference (local-server)
1~3번을 진행했다면 4-1은 건너뛰고 이후 명령어에서 본인이 finetuning한 모델 경로로 modelpath를 변경하여 진행

## 4-1. Hugging Face에서 RaccoonBot finetuned OpenVLA 모델 다운로드
서버에서 terminal에 아래 명령어를 입력하여 모델 다운로드
```
pip install -U huggingface_hub

hf download fair-lab/openvla-7b-finetuned-raccoonbot --local-dir /data/yb/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot
``` 

## 4-2. 서버측 코드 실행
server 실행 명령문<br>
만약 1~3번을 진행하여 직접 finetuning했다면 model path를 openvla-runs/ 아래에 있는 모델 디렉토리로 변경하고 진행<br>
```
cd /data/yb/Raccoonbot_Openvla/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/yb/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

## 4-3. 클라이언트측에서 실행할 환경 설정
클라이언트측 코드와 MuJoCo xml 파일 [다운로드](https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing) 후 압축 풀기 <br>
파일: openvla_multicolor_client.py, openvla_multicolor_client_real_robot.py, raccoon_env.py, Raccoon_colored_cylinder.xml, RaccoonBot_S.xml, requirements.txt

VSCode로 압축 풀은 상위 폴더를 열고 terminal에서 환경설정
```
pip install -r requirments.txt
```

## 4-4. 클라이언트측 코드 실행
target_color를 **[red, blue, green, yellow]** 로 수정하면 그에 맞게 prompt가 변경됨

⭐ local 실행 명령문
```
python openvla_multicolor_client.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer
```

## 4-5. 실제 라쿤봇을 연결하여 실행
openvla_multicolor_client_real_robot.py를 실행하면 MuJoCo 환경에서 동작하는 Action을 로봇이 동일하게 수행

⭐ local 실행 명령문
```
python openvla_multicolor_client_real_robot.py --server_url http://127.0.0.1:8000 --target_color red --use_real_robot --use_viewer
```

### 클라이언트 실행 스크립트

`Raccoonbot_Openvla` 저장소 루트에서 실행:

```bash
# MuJoCo 클라이언트
./scripts/run_sim_client.sh

# 대상 또는 서버 주소 변경
TARGET_COLOR=blue TARGET_OBJECT_TYPE=cube \
SERVER_URL=http://127.0.0.1:8000 \
./scripts/run_sim_client.sh

# 실제 로봇 클라이언트(하드웨어 실행을 명시적으로 활성화해야 함)
USE_REAL_ROBOT=1 TARGET_COLOR=red \
./scripts/run_real_robot_client.sh

# 실제 실행 없이 명령만 확인
DRY_RUN=1 ./scripts/run_sim_client.sh
DRY_RUN=1 ./scripts/run_real_robot_client.sh
```
