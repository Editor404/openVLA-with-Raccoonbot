# Raccoonbot_Openvla

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
MuJoCo 가상환경에서 finetuning을 위한 데이터를 수집
```
cd /data/Raccoonbot_Openvla/Mujoco
python raccoon_grasp_multicolor_scene_dataset.py
```

## 2. rlds 파일 변환
raw data를 rlds builder에 맞게 변경
아래 명령문 그대로 실행
```
cd /data/Raccoonbot_Openvla/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
--raw_root /data/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder \
--out_root /data/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
--val_ratio 0.1
```

## 2-1. rlds builder
rlds builder 실행
아래 명령문 그대로 실행
```
cd /data/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```
실행하면 root 하위에 tensorflow_datasets 폴더 생성됨
```
mv /root/tensorflow_datasets /data/Raccoonbot_Openvla/
```

## 3. Raccoonbot 기반 OpenVLA finetuning
아래 명령어 그대로 실행 <br>
(max_steps, save_steps 변경 가능)
```
cd /data/Raccoonbot_Openvla/openvla
export PYTHONPATH=/data/Raccoonbot_Openvla/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_grasp \
  --run_root_dir /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/openvla-runs \
  --adapter_tmp_dir /data/Raccoonbot_Openvla/raccoon_dataset/raccoon_grasp/openvla-adapter-tmp \
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

hf download fair-lab/openvla-7b-finetuned-raccoonbot --local-dir /data/openvla-runs/openvla-7b-finetuned-raccoonbot
``` 

## 4-2. 서버측 코드 실행
server 실행 명령문
```
cd /data/Raccoonbot_Openvla/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

## 4-3. 클라이언트측에서 실행할 환경 설정
클라이언트측 코드와 MuJoCo xml 파일 [다운로드](https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing) 후 압축 풀기 <br>
파일: openvla_multicolor_client.py, raccoon_env.py, Raccoon_colored_cylinder.xml, RaccoonBot_S.xml, requirements.txt

VSCode로 압축 풀은 상위 폴더를 열고 terminal에서 환경설정
```
pip install -r requirments.txt
```

## 4-4. 클라이언트측 코드 실행
target_color를 **[red, blue, green, yellow]** 로 수정하면 그에 맞게 prompt가 변경됨 <br>
⭐ local 실행 명령문
```
python openvla_multicolor_client.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer
```

