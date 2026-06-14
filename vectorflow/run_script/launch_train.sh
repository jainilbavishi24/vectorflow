export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=/home/AutoDP/jainil.bavishi/vectorflow_repo
export SAVE_DIR=/scratch3/jainil.bavishi/flowplanner_exp
export TENSORBOARD_LOG_PATH=/scratch3/jainil.bavishi/flowplanner_exp/tensorboard
export TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval
export TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json
export TORCH_LOGS="dynamic,recompiles"
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH}"

/scratch3/jainil.bavishi/flowplanner_env/bin/python -m torch.distributed.run --nnodes 1 --nproc-per-node 2 --standalone $(dirname "$0")/../trainer.py --config-name vectorflow_standard
