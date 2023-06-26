
1、安装diffusers

git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .


2、安装requirements
pip install -r requirements.txt


3、训练

3.1、accelerate的设置：
accelerate config

3.2、训练

TRAIN_DATA_DIR：图片的所在文件夹
output_dir： 保存模型的文件夹
resume_from_checkpoint：加载训练的checkpoint，然后继续训练， latest


export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/home/featurize/work/yazhou/llm_model/sd_1_4/data"

accelerate launch --mixed_precision="fp16"  pretrain.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resolution=224 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-model"


多GPU：
accelerate launch --mixed_precision="fp16" --multi_gpu  pretrain.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resolution=224 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-model"




