#!/bin/bash
export WANDB_START_METHOD=thread
export WANDB_TENSORBOARD=false
export WANDB_IGNORE_GLOBS="**/*.tfevents*,events.out.tfevents*,*.tfevents*"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python -m torch.distributed.run --standalone --nproc_per_node=6 main/rl/train.py \
    task=DexHandImitator side=RH headless=true multi_gpu=True \
    sim_device='cuda:${oc.env:LOCAL_RANK,0}' rl_device='cuda:${oc.env:LOCAL_RANK,0}' \
    num_envs=400 \
    dexhand=inspire \
    test=false \
    randomStateInit=true \
    dataIndices=all \
    learning_rate=2e-4 \
    actionsMovingAverage=0.4 \
    usePIDControl=False \
    wandb_activate=True \
