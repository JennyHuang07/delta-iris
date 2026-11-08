#!/bin/bash
# Tiny CPU run to check that the FrozenLake pipeline works end to end (NOT for timing).
# Run from the repo root:  bash scripts/smoke_frozenlake.sh
# Takes a few minutes on a laptop CPU. Outputs go to outputs/YYYY-MM-DD/hh-mm-ss/ (see timing.csv there).

python src/main.py env=frozenlake params=frozenlake \
    wandb.mode=disabled \
    params.common.device=cpu \
    params.common.epochs=2 \
    params.common.do_checkpoint=false \
    params.collection.train.num_envs=2 \
    params.collection.train.config.num_steps_first_epoch=300 \
    params.collection.train.config.num_steps=100 \
    params.collection.test.num_envs=2 \
    params.collection.test.config.num_episodes=4 \
    params.training.num_workers=0 \
    params.training.tokenizer.batch_num_samples=4 \
    params.training.tokenizer.steps_first_epoch=5 \
    params.training.tokenizer.steps_per_epoch=3 \
    params.training.world_model.batch_num_samples=4 \
    params.training.world_model.steps_first_epoch=5 \
    params.training.world_model.steps_per_epoch=3 \
    params.training.actor_critic.batch_num_samples=4 \
    params.training.actor_critic.steps_per_epoch=3 \
    params.evaluation.every=2 \
    params.evaluation.tokenizer.save_reconstructions=false \
    "$@"
