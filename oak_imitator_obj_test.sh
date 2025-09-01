#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python main/rl/train.py \
       task=DexHandImitatorEvalManip \
       dexhand=inspire \
       side=RH \
       headless=false \
       num_envs=10 \
       learning_rate=2e-4 \
       test=true \
       randomStateInit=false \
       dataIndices=all \
       actionsMovingAverage=0.4 \
       usePIDControl=False \
       wandb_activate=True \
       checkpoint=/workspace/ManipTrans/assets/imitator_rh_inspireftp.pth \
       capture_video=True \