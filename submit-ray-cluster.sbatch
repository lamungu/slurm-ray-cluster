#!/bin/bash

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J bl-multi-node

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB

#SBATCH --time=3:00:00

source /etc/profile.d/modules.sh


export TMPDIR=/home/$USER/tmp
export TMP=/home/$USER/tmp
export PIP_CACHE_DIR=/home/$USER/tmp/cache

export HF_HOME=/home/$USER/.cache/huggingface
export HF_DATASETS_CACHE=/home/$USER/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/$USER/.cache/huggingface/hub

# Load modules or your own conda environment here
module load python/3.13.2
module load cuda/12.9.1

source /home/$USER/.venv/bin/activate

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi
port=6379

echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"
srun --nodes=1 --ntasks=1 --gpus=1 -w $head_node start-head.sh $head_node_ip &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
sleep 5
##############################################################################################

#### call your code below
python examples/mnist_pytorch_trainable.py --cuda

exit
