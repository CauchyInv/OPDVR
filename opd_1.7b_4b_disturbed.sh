#!/usr/bin/env bash
# Probe-and-reweight OPD (disturbed_opd) for Setting 1: Qwen3-1.7B-Base student
# <- Qwen3-4B-Base-GRPO fixed teacher. Existing baseline/takeover scripts are
# read-only inputs and are never executed for training by this launcher.
#
# Precision follows Rethinking-OPD rather than our earlier baselines:
#   MODEL_DTYPE=fp32   -> actor/ref/teacher fsdp model_dtype, i.e. fp32 master
#                         weights with FSDP casting to bf16 for compute.
#   ROLLOUT_DTYPE=bf16 -> the vLLM engine, matching verl's own default.
# Our previous scripts drove both from one variable, so choosing bf16 for
# rollout speed silently cost the training-side master weights.
#
# Usage:
#   bash opd_1.7b_4b_disturbed.sh --cfg job          # inspect resolved config
#   KAPPA=0.0 bash opd_1.7b_4b_disturbed.sh          # kappa=0 baseline equivalence
#   ETA_PROBE=3e-4 KAPPA=0.5 bash opd_1.7b_4b_disturbed.sh
set -e -o pipefail
ulimit -c 0

DISTURBED_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DISTURBED_REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${OPDVR_CONDA_ENV:-verl}"
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DISTURBED_REPO/verl${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export WANDB_ENTITY=${WANDB_ENTITY:-camellia504-tsinghua-university}
export WANDB_DIR=${WANDB_DIR:-/data/jiangxitai/OPDVR/wandb}
# The W&B API is reachable directly from this node, and routing it through the
# shell's default mihomo proxy makes wandb.init time out. Keep the choice
# explicit and task-local; never touch the shared mihomo configuration.
if [[ "${DISTURBED_NETWORK_MODE:-direct}" == direct ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
elif [[ "${DISTURBED_NETWORK_MODE}" == proxy ]]; then
    export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7890}
    export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7890}
fi
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export VERL_USE_MODELSCOPE=${VERL_USE_MODELSCOPE:-False}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/jiangxitai/OPDVR/cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}
mkdir -p "$WANDB_DIR"

if [[ "$*" != *"--cfg"* && "$*" != *"--help"* ]]; then
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --disable-usage-stats
fi

# ---- probe-and-reweight switches ----
export DISTURBED_ENABLED=${DISTURBED_ENABLED:-True}
export ALPHA=${ALPHA:-1.0}              # 1.0 keeps the advantage identical to the baseline
export ETA_PROBE=${ETA_PROBE:-1e-4}     # calibrate over {3e-5, 1e-4, 3e-4, 1e-3}
export KAPPA=${KAPPA:-0.0}              # 0.0 must reproduce the baseline curve
export W_MAX=${W_MAX:-5.0}
export A_MIN_QUANTILE=${A_MIN_QUANTILE:-0.2}
export SELECT=${SELECT:-probe}          # probe | random | stratified_random | entropy
export STRATIFIED_BINS=${STRATIFIED_BINS:-16}
export DIRECTION=${DIRECTION:-both}     # both | up | down
export PROBE_MOMENTUM=${PROBE_MOMENTUM:-False}        # False=本批原始梯度 | True=真实Adam方向(含动量)
export PROBE_DIAGNOSTIC=${PROBE_DIAGNOSTIC:-False}   # 同参数下扫多个 eta 与两种方向，比较 V 重合度
export WEIGHT_MODE=${WEIGHT_MODE:-mean_one}          # mean_one | budget
export MASS_GROWTH_BUDGET=${MASS_GROWTH_BUDGET:-0.2} # budget mode only
export OPPOSITE_WEIGHT_SCALE=${OPPOSITE_WEIGHT_SCALE:-1.0} # 1=no-op; 0=masks opposite disturbed set
export CONTINUOUS_FLOOR=${CONTINUOUS_FLOOR:-0.25}
export CONTINUOUS_QUANTILE=${CONTINUOUS_QUANTILE:-0.9}
export CONTINUOUS_SCORE_MODE=${CONTINUOUS_SCORE_MODE:-delta} # delta | surrogate | rank | topq
export CONTINUOUS_SHUFFLE=${CONTINUOUS_SHUFFLE:-False}
export CONTINUOUS_RANK_POWER=${CONTINUOUS_RANK_POWER:-1.0}
export CONTINUOUS_TOP_FRACTION=${CONTINUOUS_TOP_FRACTION:-0.5}
export AUDIT_DIR=${AUDIT_DIR:-}
export AUDIT_POST_UPDATE=${AUDIT_POST_UPDATE:-False}
export AUDIT_TOPK=${AUDIT_TOPK:-2}
export AUDIT_DISTRIBUTION_KL_TOPK=${AUDIT_DISTRIBUTION_KL_TOPK:-0}
export AUDIT_EOS_TOKEN_ID=${AUDIT_EOS_TOKEN_ID:-151643}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}

# ---- precision, aligned with Rethinking-OPD ----
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-bfloat16}
export DATA_SHUFFLE=${DATA_SHUFFLE:-False}

# ---- OPD reward: sampled-token, no legacy gating ----
export ADV_ESTIMATOR=token_reward_direct
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
# vLLM engine seed for training rollouts; change it to get a different sampling
# stream on identical data when DATA_SHUFFLE=False; shuffle ablations override
# only this switch and preserve all other training settings.
export ROLLOUT_SEED=${ROLLOUT_SEED:-0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"}

# ---- shared hypers ----
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-256}
export N_RESPONSES=${N_RESPONSES:-1}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}

export TRAIN_DATASET=${TRAIN_DATASET:-"$DISTURBED_REPO/datasets/dapo-math-17k-train.parquet"}
export TEST_DATASET=${TEST_DATASET:-"$DISTURBED_REPO/datasets/valid_final_unique.parquet"}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/root/models/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/root/models/Qwen3-4B-Base-GRPO}

export PROJECT_NAME=${PROJECT_NAME:-curri_opd}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_1.7b_4b_disturbed_a${ALPHA}_eta${ETA_PROBE}_k${KAPPA}_$(date +%Y%m%d_%H%M%S)}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/data/jiangxitai/OPDVR/checkpoints}
export FINAL_CKPT_DIR=${FINAL_CKPT_DIR:-"$CHECKPOINT_ROOT/${EXPERIMENT_NAME}"}

MIN_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
PPO_MAX_TOKEN_LEN_PER_GPU=$(( PPO_MAX_TOKEN_LEN_PER_GPU > MIN_TOKEN_LEN_PER_GPU ? PPO_MAX_TOKEN_LEN_PER_GPU : MIN_TOKEN_LEN_PER_GPU ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.use_kl_in_reward=False \
    algorithm.disturbed_opd.enabled=$DISTURBED_ENABLED \
    algorithm.disturbed_opd.alpha=$ALPHA \
    algorithm.disturbed_opd.eta_probe=$ETA_PROBE \
    algorithm.disturbed_opd.kappa=$KAPPA \
    algorithm.disturbed_opd.w_max=$W_MAX \
    algorithm.disturbed_opd.a_min_quantile=$A_MIN_QUANTILE \
    algorithm.disturbed_opd.select=$SELECT \
    algorithm.disturbed_opd.stratified_bins=$STRATIFIED_BINS \
    algorithm.disturbed_opd.direction=$DIRECTION \
    algorithm.disturbed_opd.probe_momentum=$PROBE_MOMENTUM \
    algorithm.disturbed_opd.probe_diagnostic=$PROBE_DIAGNOSTIC \
    algorithm.disturbed_opd.weight_mode=$WEIGHT_MODE \
    algorithm.disturbed_opd.mass_growth_budget=$MASS_GROWTH_BUDGET \
    algorithm.disturbed_opd.opposite_weight_scale=$OPPOSITE_WEIGHT_SCALE \
    algorithm.disturbed_opd.continuous_floor=$CONTINUOUS_FLOOR \
    algorithm.disturbed_opd.continuous_quantile=$CONTINUOUS_QUANTILE \
    algorithm.disturbed_opd.continuous_score_mode=$CONTINUOUS_SCORE_MODE \
    algorithm.disturbed_opd.continuous_shuffle=$CONTINUOUS_SHUFFLE \
    algorithm.disturbed_opd.continuous_rank_power=$CONTINUOUS_RANK_POWER \
    algorithm.disturbed_opd.continuous_top_fraction=$CONTINUOUS_TOP_FRACTION \
    algorithm.disturbed_opd.audit_dir="$AUDIT_DIR" \
    algorithm.disturbed_opd.audit_post_update=$AUDIT_POST_UPDATE \
    algorithm.disturbed_opd.audit_topk=$AUDIT_TOPK \
    algorithm.disturbed_opd.audit_distribution_kl_topk=$AUDIT_DISTRIBUTION_KL_TOPK \
    algorithm.disturbed_opd.audit_eos_token_id=$AUDIT_EOS_TOKEN_ID \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.val_batch_size=${VAL_BATCH_SIZE:-256} \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=$DATA_SHUFFLE \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-32768} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MIN_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    +actor_rollout_ref.rollout.seed=$ROLLOUT_SEED \
    actor_rollout_ref.rollout.dtype=$ROLLOUT_DTYPE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    +actor_rollout_ref.rollout.correctness_gated=False \
    +actor_rollout_ref.rollout.grpo_scaled=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.7} \
    actor_rollout_ref.rollout.max_model_len=$MIN_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-16} \
    custom_reward_function.path="$DISTURBED_REPO/verl/verl/utils/reward_score/__init__.py" \
    custom_reward_function.name=default_compute_score \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.log_val_generations=2 \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-110} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1} \
    trainer.test_freq=${TEST_FREQ:-22} \
    trainer.total_epochs=${TOTAL_EPOCHS:-3} \
    trainer.default_local_dir="$FINAL_CKPT_DIR" "$@"
