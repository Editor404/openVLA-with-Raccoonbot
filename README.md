# RaccoonBot OpenVLA 파이프라인 확장

FAIR Lab OpenVLA 과제를 위해 RaccoonBot의 MuJoCo 데이터 생성, RLDS/TFDS
변환, OpenVLA LoRA 학습 및 추론 과정을 확장한 저장소입니다.

## 주요 구현 내용

### 데이터셋 확장

- 색상: `red`, `blue`, `green`, `yellow`
- 물체 형상: `cylinder`, `cube`, `sphere`
- 작업:
  - `grasp`
  - `push`
  - `pick_and_place`
- 작업별로 여러 자연어 명령 템플릿 지원
- `(작업, 색상, 형상)` 조합을 고려한 균형 수집
- 중단된 데이터 수집을 이어서 실행할 수 있는 resume 기능

### 데이터 품질 및 안정성 개선

- 그리퍼를 연 상태로 접근한 뒤 grasp-height에서 닫도록 수정
- 인위적인 물체 부착을 기본적으로 비활성화
- MuJoCo contact와 물리 계산을 기반으로 성공 판정
- 배치 실패 시 전체 수집을 종료하지 않고 해당 시도만 재실행
- 밀기 및 집어서 옮기기 경로에 보간 IK 사전 검사 적용
- 비정상적인 end-effector 이동량을 가진 transition 저장 방지

### OpenVLA 변환 개선

- idle frame 제거 후에도 action을 **바로 다음 원본 프레임까지의 이동량**으로 기록
- 여러 프레임의 이동량이 하나의 action으로 합쳐지는 오류 수정
- 수집기의 gripper 표현 `0=open, 1=close`를 OpenVLA 규칙
  `1=open, 0=close`로 변환
- 세 작업을 동일 확률로 학습하는 `raccoon_task_balanced` mixture 등록

## 저장소 구조

```text
Raccoonbot_Openvla/
├── Mujoco/
│   ├── raccoon_grasp_multicolor_scene_dataset.py
│   ├── raccoon_multitask_colored_objects_dataset.py
│   ├── raccoon_dataset/
│   │   ├── convert_raw_to_openvla_rlds_intermediate.py
│   │   └── prepare_task_balanced_intermediate.py
│   └── rlds_dataset_builder/
│       ├── build_task_balanced_tfds.py
│       └── raccoon_pick_place/
├── openvla/
│   ├── vla-scripts/finetune.py
│   ├── openvla_server.py
│   └── prismatic/vla/datasets/rlds/oxe/
├── scripts/
│   ├── run_checkpoint_server.sh
│   ├── run_sim_client.sh
│   └── run_real_robot_client.sh
├── results/
├── docs/
└── report.pdf
```

## 결과 요약

생성 및 변환된 task-balanced 데이터셋의 에피소드 수는 다음과 같습니다.

| 데이터셋 | 학습 | 검증 |
|---|---:|---:|
| `raccoon_grasp` | 1,080 | 120 |
| `raccoon_push` | 146 | 16 |
| `raccoon_pick_and_place` | 131 | 15 |

수정된 action label의 축별 최대 이동량은 약 4.9 mm 이하로 확인되어
클라이언트의 `max_delta_xyz=0.005` 안전 범위와 일치합니다.

### 에피소드 시각화

![초록색 정육면체 잡기 성공 에피소드](results/episode_visualizations/raccoon_grasp_episode_grid.png)

## 설치

### 시스템 패키지

```bash
sudo apt update
sudo apt install -y \
  libegl1 \
  libgl1 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libgles2 \
  libegl1-mesa \
  libegl1-mesa-dev \
  mesa-utils
```

### OpenVLA 설치

```bash
git clone git@github.com:Editor404/openVLA-with-Raccoonbot.git
cd openVLA-with-Raccoonbot/openvla
pip install .
```

MuJoCo 데이터 생성과 TFDS 빌드에는 해당 환경의 `mujoco`,
`tensorflow`, `tensorflow-datasets` 패키지도 필요합니다.

## 1. MuJoCo 데이터 생성

### 잡기 데이터

기본 설정은 네 색상과 세 형상에 대해 총 1,200개의 성공 에피소드를
수집합니다.

```bash
cd Mujoco
python3 raccoon_grasp_multicolor_scene_dataset.py
```

기본 출력 경로:

```text
Mujoco/raccoon_grasp_colored_objects/
```

수집 개수는 스크립트 마지막의 `num_episodes` 인자로 변경할 수 있습니다.

### 밀기 및 집어서 옮기기 데이터

기본 설정은 두 작업의 색상·형상 조합에 대해 총 720개의 성공
에피소드를 수집합니다.

```bash
cd Mujoco
python3 raccoon_multitask_colored_objects_dataset.py
```

기본 출력 경로:

```text
Mujoco/raccoon_multitask_colored_objects/
```

기존 성공 에피소드가 있으면 조합별 개수와 다음 episode ID를 복구하여
이어서 수집합니다.

## 2. RLDS intermediate 변환

세 작업을 각각의 intermediate 데이터셋으로 변환합니다.

```bash
cd Mujoco/raccoon_dataset
python3 prepare_task_balanced_intermediate.py \
  2>&1 | tee ../../results/logs/intermediate_conversion.log
```

생성 경로:

```text
Mujoco/raccoon_dataset/task_balanced_intermediate/
├── raccoon_grasp/
├── raccoon_push/
└── raccoon_pick_and_place/
```

이미지는 불필요한 복사를 줄이기 위해 기본적으로 hard link 방식으로
연결됩니다.

## 3. TFDS 빌드

```bash
cd Mujoco/rlds_dataset_builder
python3 build_task_balanced_tfds.py --overwrite \
  2>&1 | tee ../../results/logs/tfds_build.log
```

생성 경로:

```text
tensorflow_datasets/
├── raccoon_grasp/
├── raccoon_push/
└── raccoon_pick_and_place/
```

특정 데이터셋만 빌드하려면:

```bash
python3 build_task_balanced_tfds.py \
  --datasets raccoon_grasp raccoon_push \
  --overwrite
```

## 4. OpenVLA LoRA 학습

학습용 GPU 서버에서 실행합니다. OpenVLA 7B LoRA 학습에는 A100과 같은
충분한 VRAM의 GPU 사용을 권장합니다.

```bash
cd openvla
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../tensorflow_datasets \
  --dataset_name raccoon_task_balanced \
  --run_root_dir ./openvla-runs \
  --adapter_tmp_dir ./openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --max_steps 100 \
  --save_steps 100 \
  --image_aug False \
  --run_id_note raccoon-task-balanced
```

`max_steps=100`은 파이프라인 확인용 짧은 실행입니다. 최종 성능 평가를
위해서는 더 긴 학습과 작업별 검증이 필요합니다.

기존 노트북의 A100 학습 기록은 `results/logs/training.log`에 포함되어
있지만, 새 `raccoon_task_balanced` mixture의 전체 학습은 별도로
수행해야 합니다.

## 5. 추론 서버 실행

체크포인트는 저장소 외부에 배치합니다.

```bash
CHECKPOINT_DIR=/path/to/openvla-checkpoint \
./scripts/run_checkpoint_server.sh
```

경로와 실행 명령만 확인하려면:

```bash
DRY_RUN=1 \
CHECKPOINT_DIR=/path/to/openvla-checkpoint \
./scripts/run_checkpoint_server.sh
```

직접 실행하는 경우:

```bash
cd openvla
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python3 openvla_server.py \
  --model_path /path/to/openvla-checkpoint \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

task-balanced 학습 체크포인트를 사용할 때는 checkpoint의
`dataset_statistics.json`에 저장된 normalization key를 확인해야 합니다.

## 6. 원격 서버 연결

서버의 8000 포트가 외부에 공개되지 않은 경우 SSH 터널을 사용합니다.

```bash
ssh -L 8000:127.0.0.1:8000 USER@HOST -p PORT
```

터널이 열린 동안 로컬 클라이언트는 다음 주소를 사용합니다.

```text
http://127.0.0.1:8000
```

## 7. MuJoCo 클라이언트 실행

클라이언트 파일은 저장소 상위의 `executeCode/`에 있다고 가정합니다.

```bash
./scripts/run_sim_client.sh
```

색상과 형상을 지정하려면:

```bash
TARGET_COLOR=green \
TARGET_OBJECT_TYPE=cube \
SERVER_URL=http://127.0.0.1:8000 \
./scripts/run_sim_client.sh
```

실제 실행 없이 명령만 확인:

```bash
DRY_RUN=1 ./scripts/run_sim_client.sh
```

## 8. 실제 RaccoonBot 실행

실제 로봇이 움직이므로 작업 공간과 비상 정지 상태를 먼저 확인해야
합니다. 실수로 실행되지 않도록 `USE_REAL_ROBOT=1`을 명시해야 합니다.

```bash
USE_REAL_ROBOT=1 \
TARGET_COLOR=red \
./scripts/run_real_robot_client.sh
```

명령 확인:

```bash
DRY_RUN=1 ./scripts/run_real_robot_client.sh
```

## 테스트

MuJoCo 수집 및 변환 회귀 테스트:

```bash
PYTHONPATH=Mujoco \
python3 -m unittest discover -s Mujoco -p 'test_*.py' -v
```

OpenVLA 등록 및 TFDS 빌드 테스트:

```bash
python3 -m unittest -v \
  Mujoco.rlds_dataset_builder.test_openvla_task_balanced_registration \
  Mujoco.rlds_dataset_builder.test_task_balanced_tfds_build
```

생성된 intermediate 데이터가 없는 clone 환경에서는 실제 manifest 개수
검사가 의도적으로 skip됩니다.

## 제출 결과물

| 경로 | 내용 |
|---|---|
| `report.pdf` | 한국어 과제 보고서 |
| `docs/short_report.md` | 보고서 원문 |
| `docs/assignment_worklog.md` | 구현 과정 기록 |
| `results/episode_visualizations/` | 에피소드 시각화 |
| `results/logs/tfds_build.log` | TFDS 빌드 기록 |
| `results/logs/training.log` | 기존 A100 학습 기록 발췌 |
| `results/logs/inference_server.log` | 추론 서버 및 action 출력 발췌 |

## Git에 포함하지 않는 파일

과제 지침과 저장소 용량 제한에 따라 다음 파일은 커밋하지 않습니다.

- raw MuJoCo 이미지 데이터
- RLDS intermediate 데이터
- TFRecord 및 `tensorflow_datasets/`
- LoRA adapter 및 학습 checkpoint
- `*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`
- 데이터 전송용 대용량 압축파일

이 파일들은 위 명령으로 재생성하거나 별도 스토리지에서 관리해야 합니다.

## 참고

- 과제 보고서: [`report.pdf`](report.pdf)
- 상세 작업 기록: [`docs/assignment_worklog.md`](docs/assignment_worklog.md)
- 원본 FAIR Lab 저장소:
  <https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla>
