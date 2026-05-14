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


redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]}
ip=$node_1
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 shifter \
  bash multi-node-serving.sh leader --ray_port=6379 --ray_cluster_size=$SLURM_JOB_NUM_NODES && \
  vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --port 8080 --tensor-parallel-size 8 --pipeline_parallel_size 2 &
sleep 30

# On the head node machine, start the Ray head node process and run a vLLM server.
#   ./multi-node-serving.sh leader --ray_port=6379 --ray_cluster_size=<SIZE> [<extra ray args>]  && \
#   vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --port 8080 --tensor-parallel-size 8 --pipeline_parallel_size 2
# 
# On each worker node, start the Ray worker node process.
#   ./multi-node-serving.sh worker --ray_address=<HEAD_NODE_IP> --ray_port=6379 [<extra ray args>]
#
worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i shifter \
  bash ./multi-node-serving.sh worker --ray_address=$ip --ray_port=6379 &
  sleep 5
done
##############################################################################################

#### stay here indefinitely (until it expires)
wait


