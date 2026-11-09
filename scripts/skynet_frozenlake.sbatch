#!/bin/bash
# Delta-IRIS on FrozenLake, timing run on Skynet (Slurm).
# Submit from the repo root:   sbatch scripts/skynet_frozenlake.sbatch
# Check status:                squeue -u $USER
# Follow the log:              tail -f slurm-delta-iris-fl-<jobid>.out   (ignored by .gitignore)
#
# ei-lab partition: max 4h per job, not preempted. timing.csv is written after every epoch,
# so if the job hits the 4h limit, all finished epochs are still logged.

#SBATCH --job-name=delta-iris-fl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --partition=ei-lab
#SBATCH --account=ei-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1           # needs a bf16-capable GPU (Ampere or newer: a40 / a5000 / l40s). rtx_6000, 2080_ti, titan_x do NOT support bf16
#SBATCH --cpus-per-task=8          # 4 DataLoader workers + 16 env processes are light; 8 cores is plenty
#SBATCH --mem=32G
#SBATCH --time=4:00:00             # ei-lab partition limit

set -eo pipefail   # no -u: conda's activation scripts reference unset variables

# Load conda directly: ~/.bashrc returns early in non-interactive (batch) shells, so `conda init` has no effect here
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate delta-iris

# Flush Python output immediately so the Slurm log (and `tail -f`) shows progress live instead of in delayed chunks
export PYTHONUNBUFFERED=1

# Record the hardware so timing numbers can be interpreted later
echo "host: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())"

# wandb: use online if the compute node has internet, otherwise offline and `wandb sync` later from the login node
python src/main.py env=frozenlake params=frozenlake \
    wandb.mode=${WANDB_MODE:-offline} \
    wandb.name=frozenlake-timing-${SLURM_JOB_ID} \
    "$@"
