#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python main/rl/train.py \
       task=DexHandImitator \
       dexhand=inspire \
       side=RH \
       headless=false \
       num_envs=4 \
       learning_rate=2e-4 \
       test=true \
       randomStateInit=false \
       dataIndices=[20aed@0,e9aab@2,925aa@1] \
       actionsMovingAverage=0.4 \
       usePIDControl=True \
       wandb_activate=True \
       capture_video=True \
       checkpoint=/workspace/ManipTrans/runs/DexHandImitator__08-20-06-13-55/nn/last_DexHandImitator_ep_53000_rew_208.61267_sr_0.7050619125366211_fr_0.29493772983551025.pth
    #    save_video=True \
