# This script took the best checkpoints trained on the ~200 training samples,

# Which are converted into huggingface format 
#  and try to continue training them
# on the rest of the annoated dataset 

set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
# i divided all batch size by 2. 
# change son june
# 
# learning rate 1e-5
# use the repcount data to train (1024)
# validate on the orgional repcount data
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/alexli/data/repcount/og/train_single_prompt.parquet \
    data.val_files=/home/alexli/data/repcount/val_single_prompt.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=25000 \
    data.max_response_length=6000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.video_key=videos \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_repcount_og' \
    trainer.experiment_name='qwen2_5_vl_3b_repcount_og_single_detailed_prompt' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@