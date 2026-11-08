#!/bin/bash
# Delta-IRIS on FrozenLake, timing run on Skynet (Slurm).
# Submit from the repo root:   sbatch scripts/skynet_frozenlake.sbatch
# Check status:                squeue -u $USER
# Follow the log:              tail -f slurm-delta-iris-fl-<jobid>.out   (ignored by .gitignore)
#
# TODO: fill in partition / account / qos according to the lab's Skynet guide (HCAI-Lab-GT/skynet).

#SBATCH --job-name=delta-iris-fl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8          # 4 DataLoader workers + 16 env processes are light; 8 cores is plenty
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

source ~/.bashrc
conda activate delta-iris

# Record the hardware so timing numbers can be interpreted later
echo "host: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"

# wandb: use online if the compute node has internet, otherwise offline and `wandb sync` later from the login node
python src/main.py env=frozenlake params=frozenlake \
    wandb.mode=${WANDB_MODE:-offline} \
    wandb.name=frozenlake-timing-${SLURM_JOB_ID} \
    "$@"
