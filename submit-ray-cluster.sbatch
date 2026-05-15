#!/encs/bin/bash

#SBATCH -o slurm_logs/%x_%j_master.out
#SBATCH -e slurm_logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J bl-multi-node
#SBATCH --time=03:00:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus=2

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB


SCRATCH_DIR="/speed-scratch/$USER"

# Set all the temporary directories
export TMPDIR=$SCRATCH_DIR/tmp
export TMP=$SCRATCH_DIR/tmp
export PIP_CACHE_DIR=$SCRATCH_DIR/tmp/cache
export HF_HOME=$SCRATCH_DIR/.cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH_DIR/.cache/huggingface/datasets
export HF_HUB_CACHE=$SCRATCH_DIR/.cache/huggingface/hub

# Load modules or your own conda environment here
module load python/3.12.0/default
module load cuda/12.8/default

source $SCRATCH_DIR/.vllm-venv/bin/activate

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

# Run the leader node
srun --nodes=1 --ntasks=1 -w $head_node ./multi-node-serving.sh leader --ray_port=$port --ray_cluster_size=$SLURM_JOB_NUM_NODES && \
  vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 2 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node ./multi-node-serving.sh worker --ray_address=$head_node_ip --ray_port=$port

##############################################################################################
# wait indefinitely until the job dies
wait