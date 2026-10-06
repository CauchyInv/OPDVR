# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        forward_dtype = self.config.get("forward_autocast_dtype", "bf16")
        if forward_dtype not in ("bf16", "fp32"):
            raise ValueError("forward_autocast_dtype must be bf16 or fp32")
        with torch.autocast(
            device_type=self.device_name,
            dtype=torch.bfloat16,
            enabled=forward_dtype == "bf16",
        ):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            target_temperatures = None
            if "takeover_suffix_sft_mask" in micro_batch:
                from verl.trainer.ppo.teacher_takeover import prediction_temperatures

                if self.use_fused_kernels or self.use_ulysses_sp or top_k:
                    raise ValueError("Takeover v1 requires unfused sampled-token logits and SP=1")
                target_temperatures = prediction_temperatures(
                    input_ids, micro_batch["takeover_suffix_sft_mask"], temperature
                )
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    if target_temperatures is None:
                        logits_rmpad.div_(temperature)
                    else:
                        logits_rmpad.div_(target_temperatures.reshape(-1)[indices].unsqueeze(-1))

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    if target_temperatures is None:
                        logits.div_(temperature)
                    else:
                        logits.div_(target_temperatures.unsqueeze(-1))
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _clip_grad_norm(self):
        """Clip gradients in place and return the pre-clip norm, without stepping.

        Extracted so the disturbed-OPD probe reuses exactly the clipping the real
        update applies.
        """
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        return grad_norm

    def _optimizer_step(self):
        grad_norm = self._clip_grad_norm()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        if "disturbed_opd" in data.meta_info:
            return self._update_disturbed_opd_policy(data)
        if "teacher_takeover" in data.meta_info:
            return self._update_takeover_policy(data)
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    def _update_takeover_policy(self, data: DataProto):
        """One logical question batch, one update, irrespective of trajectory count."""
        from verl.trainer.ppo.teacher_takeover import LOSS_NAMES, MASK_KEYS, component_sums, normalized_loss

        if self.config.ppo_epochs != 1 or self.config.use_kl_loss or self.config.entropy_coeff != 0:
            raise ValueError("Takeover v1 supports one PPO epoch and no auxiliary entropy/reference loss")
        self.actor_module.train()
        keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids",
                "old_log_probs", "advantages", *MASK_KEYS]
        data = data.select(batch_keys=keys, non_tensor_batch_keys=[])
        device = get_device_id()
        counts = torch.stack([data.batch[key].sum() for key in MASK_KEYS]).to(device=device, dtype=torch.float64)
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        if distributed:
            torch.distributed.all_reduce(counts)
        if self.config.use_dynamic_bsz:
            micro_batches, _ = prepare_dynamic_batch(data, max_token_len=self.config.ppo_max_token_len_per_gpu)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
        weights = data.meta_info["teacher_takeover"]["weights"]
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        self.actor_optimizer.zero_grad()
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)
            inputs = dict(micro_batch.batch)
            _, log_probs, *_ = self._forward_micro_batch(inputs, temperature=data.meta_info["temperature"])
            sums = component_sums(log_probs, inputs["old_log_probs"], inputs["advantages"],
                                  [inputs[key] for key in MASK_KEYS], self.config)
            loss = normalized_loss(sums, counts, weights, world_size)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite takeover loss; refusing optimizer update")
            loss.backward()
            totals += sums.detach().double()
        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite takeover gradient; optimizer step was skipped")
        if distributed:
            torch.distributed.all_reduce(totals)
        means = totals / counts.clamp(min=1)
        metrics = {"actor/grad_norm": float(grad_norm.detach()), "takeover/optimizer_steps": 1.0,
                   "actor/pg_loss": float((means * means.new_tensor(weights)).sum())}
        for i, name in enumerate(LOSS_NAMES):
            metrics[f"takeover/loss_{name}"] = float(means[i])
            metrics[f"takeover/tokens_{name}"] = float(counts[i])
        return metrics

    @staticmethod
    def _local_view(tensor):
        """Local shard of a possibly-sharded parameter, for elementwise bookkeeping."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _disturbed_advantages(self, inputs, alpha):
        """A_t from the sampled-token log ratio.

        The baseline's `advantages` already equals the token-level distillation
        reward (token_reward_direct applies no normalization), so alpha=1 returns
        it untouched and the update differs from the baseline only by weighting.
        When rm_scores is also present the two are cross-checked, because a
        silent mismatch here would corrupt every advantage.
        """
        from verl.trainer.ppo.disturbed_opd import advantage_from_logratio

        advantages = inputs["advantages"]
        mask = inputs["response_mask"].bool()
        if "rm_scores" in inputs:
            gap = ((advantages - inputs["rm_scores"]) * mask).abs().max()
            if not bool(torch.isfinite(gap)) or float(gap) > 1e-3:
                raise ValueError(
                    "advantages and rm_scores disagree on valid tokens "
                    f"(max |gap| = {float(gap):.3e}); disturbed OPD assumes the "
                    "advantage is the raw teacher-minus-student log ratio"
                )
        if alpha == 1:
            return advantages.detach()
        return advantage_from_logratio(advantages.detach(), alpha).to(advantages.dtype)

    def _update_disturbed_opd_policy(self, data: DataProto):
        """Probe a uniform-weight step, then compensate the tokens it pushed back.

        Four passes over the same micro-batch partition: gradient probe, delta
        apply, perturbed forward (parameters restored bit-exactly afterwards),
        and the single real update.
        """
        from verl.trainer.ppo.disturbed_opd import (
            DIAG_KEYS,
            DisturbedOPDConfig,
            abs_histogram,
            attenuate_opposite_weights,
            budget_lambda,
            continuous_rank_scores,
            continuous_raw_severity,
            continuous_scores_from_severity,
            entropy_mask,
            lambda_from_sums,
            probe_delta,
            probe_mask,
            random_mask,
            selection_diagnostics,
            stratified_random_masks,
            stratified_shuffle_scores,
            threshold_from_histogram,
            token_weights,
            token_weights_budget,
            token_weights_continuous_budget,
            top_fraction_mask,
            topk_tail_forward_kl,
            uniform_weights,
            weighted_policy_sum,
        )

        cfg = DisturbedOPDConfig.from_config(data.meta_info["disturbed_opd"])
        if cfg.method != "reweight":
            from verl.trainer.ppo.disturbed_updates import update_disturbed_v5

            return update_disturbed_v5(self, data, cfg)
        if self.config.ppo_epochs != 1 or self.config.use_kl_loss or self.config.entropy_coeff != 0:
            raise ValueError("Disturbed OPD v1 supports one PPO epoch and no auxiliary entropy/reference loss")
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        device = get_device_id()
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1

        keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids",
                "old_log_probs", "advantages"]
        if "audit_response_id" in data.batch:
            keys.append("audit_response_id")
        if "rm_scores" in data.batch:
            keys.append("rm_scores")
        if audit_enabled := bool(cfg.audit_dir):
            keys.extend(
                key
                for key in ("teacher_top_k_ids", "teacher_top_k_log_probs", "teacher_entropy")
                if key in data.batch
            )
        data = data.select(batch_keys=keys, non_tensor_batch_keys=[])

        totals = torch.zeros(1, dtype=torch.float64, device=device)
        totals[0] = data.batch["response_mask"].sum()
        if distributed:
            torch.distributed.all_reduce(totals)
        valid_count = float(totals[0])
        if valid_count <= 0:
            raise ValueError("No valid response tokens in this batch")

        if self.config.use_dynamic_bsz:
            micro_batches, _ = prepare_dynamic_batch(data, max_token_len=self.config.ppo_max_token_len_per_gpu)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
        micro_batches = [micro_batch.to(device) for micro_batch in micro_batches]
        need_entropy = cfg.select == "entropy" or audit_enabled
        audit_kl_topk = cfg.audit_distribution_kl_topk
        if audit_kl_topk:
            for key in ("teacher_top_k_ids", "teacher_top_k_log_probs"):
                if key not in data.batch:
                    raise ValueError(f"distribution KL audit requires {key}")
            if data.batch["teacher_top_k_ids"].shape[-1] != audit_kl_topk:
                raise ValueError("teacher top-K width differs from audit_distribution_kl_topk")

        # ---- pass 1: uniform-weight gradient, which is what the probe perturbs by ----
        timings = {}
        started = time.monotonic()
        self.actor_optimizer.zero_grad()
        cached = []
        histogram = torch.zeros(cfg.quantile_bins + 1, dtype=torch.int64, device=device)
        for micro_batch in micro_batches:
            inputs = dict(micro_batch.batch)
            teacher_ids = inputs.get("teacher_top_k_ids") if audit_kl_topk else None
            requested_topk = audit_kl_topk if audit_kl_topk else (cfg.audit_topk if audit_enabled else 0)
            entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                inputs,
                temperature=temperature,
                calculate_entropy=need_entropy,
                top_k=requested_topk,
                student_top_k_ids=teacher_ids,
            )
            mask = inputs["response_mask"].bool()
            advantages = self._disturbed_advantages(inputs, cfg.alpha)
            total = weighted_policy_sum(log_probs, advantages, uniform_weights(mask), mask)
            loss = -total / valid_count * world_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite probe loss; refusing optimizer update")
            loss.backward()
            histogram += abs_histogram(advantages, mask, cfg.quantile_bins, cfg.quantile_max, cfg.quantile_min)
            distribution_kl_before = None
            if audit_kl_topk:
                distribution_kl_before = topk_tail_forward_kl(
                    inputs["teacher_top_k_log_probs"], topk_log_probs
                )
            cached.append({
                "advantages": advantages,
                "log_probs": log_probs.detach(),
                "mask": mask,
                "entropy": entropy.detach() if entropy is not None else None,
                # The legacy student-top-K columns remain opt-in.  KL audit
                # gathers student probabilities on teacher IDs instead and
                # stores only the scalar KL, avoiding enormous K-wide parquet.
                "topk_ids_before": (
                    topk_ids.detach() if topk_ids is not None and not audit_kl_topk else None
                ),
                "topk_log_probs_before": (
                    topk_log_probs.detach() if topk_log_probs is not None and not audit_kl_topk else None
                ),
                "distribution_kl_before": distribution_kl_before,
                "teacher_entropy": inputs.get("teacher_entropy"),
            })
        probe_norm = self._clip_grad_norm()
        timings["probe_backward"] = time.monotonic() - started
        if not torch.isfinite(probe_norm):
            self.actor_optimizer.zero_grad()
            print("WARN: non-finite probe gradient; skipping this disturbed-OPD update")
            return {"disturbed/skipped_nonfinite_grad": 1.0, "actor/grad_norm": float(probe_norm)}

        if distributed:
            torch.distributed.all_reduce(histogram)
        threshold = threshold_from_histogram(histogram, cfg.a_min_quantile, cfg.quantile_max, cfg.quantile_min)

        # ---- pass 2: apply the probe delta, keeping an exact backup ----
        started = time.monotonic()
        group = self.actor_optimizer.param_groups[0]
        beta1 = float(group["betas"][0])
        beta2 = float(group["betas"][1])
        eps = float(group["eps"])

        def apply_probe(eta, use_momentum):
            """Apply a probe delta and return (backup, moved, counted)."""
            backups, moved, counted = {}, 0, 0
            for param_group in self.actor_optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is None:
                        continue
                    state = self.actor_optimizer.state.get(param, {})
                    step_index = int(state["step"]) + 1 if "step" in state else 1
                    delta = probe_delta(
                        param.grad, state.get("exp_avg_sq"), step_index, eta, beta2, eps,
                        exp_avg=state.get("exp_avg"), use_momentum=use_momentum, beta1=beta1,
                    )
                    backups[param] = param.data.clone()
                    param.data.add_(delta.to(param.dtype))
                    local_before = self._local_view(backups[param])
                    local_after = self._local_view(param.data)
                    moved += int((local_before != local_after).sum())
                    counted += local_before.numel()
            return backups, moved, counted

        def measure_probe(threshold_value):
            """Forward at the perturbed parameters; return (sums, per-mb V masks)."""
            sums = torch.zeros(3, dtype=torch.float64, device=device)
            masks = []
            with torch.no_grad():
                for micro_batch, entry in zip(micro_batches, cached, strict=True):
                    _, shifted, *_ = self._forward_micro_batch(
                        dict(micro_batch.batch), temperature=temperature
                    )
                    change = (shifted - entry["log_probs"]).to(torch.float32)
                    selected = probe_mask(
                        entry["advantages"], change, entry["mask"], threshold_value, cfg.direction
                    )
                    masks.append((change, selected))
                    product = (entry["advantages"].to(torch.float32) * change) * entry["mask"]
                    sums[0] += product.sum().double()
                    sums[1] += (product * selected).sum().double()
                    sums[2] += selected.sum().double()
            return sums, masks

        def restore(backups):
            for param, backup in backups.items():
                param.data.copy_(backup)
            backups.clear()

        diagnostic_metrics = {}
        diagnostic_tags = []
        if cfg.probe_diagnostic:
            # Same parameters, same data, same micro-batch packing for every combo,
            # so V differences are attributable to the probe itself.
            reference = None
            diagnostic_grid = [
                (1e-6, False),
                (3e-6, False),
                (1e-5, False),
                (3e-5, False),
                (1e-4, False),
                (3e-4, False),
                (1e-3, False),
                (cfg.eta_probe, True),
            ]
            for eta, use_mom in diagnostic_grid:
                tag = f"{'mom' if use_mom else 'grad'}_eta{eta:g}"
                backups, moved, counted = apply_probe(eta, use_mom)
                sums, masks = measure_probe(threshold)
                restore(backups)
                if audit_enabled:
                    diagnostic_tags.append(tag)
                    for entry, (change, _) in zip(cached, masks, strict=True):
                        entry[f"diag/{tag}/delta"] = change
                        entry[f"diag/{tag}/up"] = probe_mask(
                            entry["advantages"], change, entry["mask"], threshold, "up"
                        )
                        entry[f"diag/{tag}/down"] = probe_mask(
                            entry["advantages"], change, entry["mask"], threshold, "down"
                        )
                if distributed:
                    torch.distributed.all_reduce(sums)
                g_, d_, n_ = float(sums[0]), float(sums[1]), float(sums[2])
                lam_max = g_ / abs(d_) if d_ else float("inf")
                diagnostic_metrics[f"diag/{tag}/vfrac"] = n_ / valid_count
                diagnostic_metrics[f"diag/{tag}/lambda_max"] = lam_max if math.isfinite(lam_max) else -1.0
                diagnostic_metrics[f"diag/{tag}/gain"] = g_
                diagnostic_metrics[f"diag/{tag}/moved_frac"] = moved / max(counted, 1)
                flat = torch.cat([m[1].reshape(-1) for m in masks])
                if reference is None:
                    reference, reference_tag = flat, tag
                else:
                    overlap = torch.zeros(2, dtype=torch.float64, device=device)
                    overlap[0] = (flat & reference).sum()
                    overlap[1] = (flat | reference).sum()
                    if distributed:
                        torch.distributed.all_reduce(overlap)
                    diagnostic_metrics[f"diag/{tag}/iou_vs_{reference_tag}"] = float(
                        overlap[0] / overlap[1].clamp(min=1)
                    )
        backups, moved, counted = apply_probe(cfg.eta_probe, cfg.probe_momentum)
        timings["probe_apply"] = time.monotonic() - started

        # ---- pass 3: perturbed forward, then restore bit-exactly ----
        started = time.monotonic()
        self.actor_optimizer.zero_grad()
        sums, masks = measure_probe(threshold)
        for entry, (change, selected) in zip(cached, masks, strict=True):
            entry["change"] = change
            entry["selected"] = selected
            if audit_enabled:
                entry["audit_up"] = probe_mask(
                    entry["advantages"], change, entry["mask"], threshold, "up"
                )
                entry["audit_down"] = probe_mask(
                    entry["advantages"], change, entry["mask"], threshold, "down"
                )
        restore(backups)
        timings["probe_forward"] = time.monotonic() - started

        if distributed:
            torch.distributed.all_reduce(sums)
        gain, damage, selected_count = float(sums[0]), float(sums[1]), float(sums[2])

        # The attenuation ablations keep the primary selector/lambda exactly
        # unchanged and independently identify the disturbed set of the
        # opposite advantage sign from the same raw probe.
        opposite_count_tensor = torch.zeros((), dtype=torch.float64, device=device)
        opposite_direction = "down" if cfg.direction == "up" else "up"
        for entry in cached:
            if cfg.opposite_weight_scale < 1.0:
                entry["opposite_selected"] = probe_mask(
                    entry["advantages"], entry["change"], entry["mask"], threshold, opposite_direction
                )
                opposite_count_tensor += entry["opposite_selected"].sum().double()
            else:
                entry["opposite_selected"] = torch.zeros_like(entry["mask"], dtype=torch.bool)
        if distributed:
            torch.distributed.all_reduce(opposite_count_tensor)
        opposite_count = float(opposite_count_tensor)

        # Matched-rate ablations reuse the probe's size but not its choice, so the
        # damage term is recomputed for the set actually being boosted.
        if cfg.select != "probe":
            rate = selected_count / valid_count if valid_count > 0 else 0.0
            alt = torch.zeros(2, dtype=torch.float64, device=device)
            if cfg.select == "entropy":
                entropy_hist = torch.zeros(cfg.quantile_bins + 1, dtype=torch.int64, device=device)
                for entry in cached:
                    entropy_hist += abs_histogram(entry["entropy"], entry["mask"], cfg.quantile_bins, cfg.quantile_max, cfg.quantile_min)
                if distributed:
                    torch.distributed.all_reduce(entropy_hist)
                entropy_threshold = threshold_from_histogram(entropy_hist, max(0.0, 1.0 - rate), cfg.quantile_max, cfg.quantile_min)
            generator = torch.Generator(device=device)
            rank = torch.distributed.get_rank() if distributed else 0
            generator.manual_seed(
                cfg.random_seed + int(data.meta_info.get("global_steps", 0)) + rank * 1_000_003
            )
            if cfg.select == "stratified_random":
                random_masks = stratified_random_masks(
                    [entry["advantages"] for entry in cached],
                    [entry["mask"] for entry in cached],
                    [entry["selected"] for entry in cached],
                    cfg.stratified_bins,
                    cfg.quantile_min,
                    cfg.quantile_max,
                    generator,
                )
            else:
                random_masks = [None] * len(cached)
            overlap = torch.zeros(2, dtype=torch.float64, device=device)
            for entry, strict_mask in zip(cached, random_masks, strict=True):
                probe_selected = entry["selected"]
                if cfg.select == "random":
                    entry["selected"] = random_mask(entry["mask"], rate, generator)
                elif cfg.select == "stratified_random":
                    entry["selected"] = strict_mask
                else:
                    entry["selected"] = entropy_mask(entry["entropy"], entry["mask"], entropy_threshold)
                product = (entry["advantages"].to(torch.float32) * entry["change"]) * entry["mask"]
                alt[0] += (product * entry["selected"]).sum().double()
                alt[1] += entry["selected"].sum().double()
                overlap[0] += (probe_selected & entry["selected"]).sum().double()
                overlap[1] += (probe_selected | entry["selected"]).sum().double()
            if distributed:
                torch.distributed.all_reduce(alt)
                torch.distributed.all_reduce(overlap)
            diagnostic_metrics["disturbed/selector_actual_product"] = float(alt[0])
            diagnostic_metrics["disturbed/selector_iou_vs_probe"] = float(
                overlap[0] / overlap[1].clamp(min=1)
            )
            if cfg.select == "stratified_random":
                # Strict control: token identities change, but intervention
                # size and lambda remain exactly those of the real probe mask.
                diagnostic_metrics["disturbed/selector_count_error"] = float(alt[1]) - selected_count
            else:
                damage, selected_count = float(alt[0]), float(alt[1])

        continuous_metrics = {}
        severity_hist = None
        reference_target = None
        if cfg.weight_mode == "continuous_budget":
            # Build raw severity once. Delta/rank/top-q use opposed log-prob
            # displacement; surrogate uses the exact frozen-advantage token
            # loss increase. Only selected entries contribute to the histogram.
            severity_hist = torch.zeros(cfg.quantile_bins + 1, dtype=torch.int64, device=device)
            for entry in cached:
                entry["raw_severity"] = continuous_raw_severity(
                    entry["advantages"], entry["change"], cfg.continuous_score_mode
                )
                severity_hist += abs_histogram(
                    entry["raw_severity"],
                    entry["selected"],
                    cfg.quantile_bins,
                    cfg.quantile_max,
                    cfg.quantile_min,
                )
            if distributed:
                torch.distributed.all_reduce(severity_hist)

            if cfg.continuous_score_mode == "topq":
                # Preserve the target mass that the original delta-continuous
                # rule would have requested on this exact probe mask. The top-q
                # subset receives no more mass than that reference and may get
                # less if w_max or kappa makes it impossible.
                reference_scale = max(
                    threshold_from_histogram(
                        severity_hist,
                        cfg.continuous_quantile,
                        cfg.quantile_max,
                        cfg.quantile_min,
                    ),
                    cfg.quantile_min,
                )
                for entry in cached:
                    entry["reference_continuous_score"] = continuous_scores_from_severity(
                        entry["raw_severity"],
                        entry["selected"],
                        reference_scale,
                        cfg.continuous_floor,
                    )
                reference_lam, _, _ = lambda_from_sums(gain, damage, cfg.kappa)
                reference_capacity = torch.zeros((), dtype=torch.float64, device=device)
                for entry in cached:
                    reference_capacity += (
                        reference_lam * entry["reference_continuous_score"].double()
                    ).clamp(max=cfg.w_max - 1).sum()
                if distributed:
                    torch.distributed.all_reduce(reference_capacity)
                reference_target = min(cfg.mass_growth_budget * valid_count, float(reference_capacity))

                top_threshold = threshold_from_histogram(
                    severity_hist,
                    1.0 - cfg.continuous_top_fraction,
                    cfg.quantile_max,
                    cfg.quantile_min,
                )
                top_sums = torch.zeros(2, dtype=torch.float64, device=device)
                for entry in cached:
                    entry["selected"] = top_fraction_mask(
                        entry["raw_severity"], entry["selected"], top_threshold
                    )
                    product = (entry["advantages"].float() * entry["change"]) * entry["mask"]
                    top_sums[0] += (product * entry["selected"]).sum().double()
                    top_sums[1] += entry["selected"].sum().double()
                if distributed:
                    torch.distributed.all_reduce(top_sums)
                damage, selected_count = float(top_sums[0]), float(top_sums[1])
                continuous_metrics["continuous/top_threshold"] = top_threshold
                continuous_metrics["continuous/reference_target_mass_growth"] = reference_target / valid_count

        lam, lam_max, fell_back = lambda_from_sums(gain, damage, cfg.kappa)
        lam_uncapped = lam
        v_fraction = selected_count / valid_count if valid_count > 0 else 0.0
        if cfg.weight_mode == "budget":
            # Non-selected tokens keep weight 1 here, so bound the extra mass
            # instead of paying for the boost by attenuating the majority.
            lam = budget_lambda(lam, v_fraction, cfg.mass_growth_budget)
        elif cfg.weight_mode == "continuous_budget":
            if cfg.continuous_score_mode == "rank":
                severity_scale = -1.0
                for entry in cached:
                    entry["continuous_score"] = continuous_rank_scores(
                        entry["raw_severity"],
                        entry["selected"],
                        severity_hist,
                        cfg.quantile_max,
                        cfg.quantile_min,
                        cfg.continuous_floor,
                        cfg.continuous_rank_power,
                    )
            elif cfg.continuous_score_mode == "topq":
                severity_scale = continuous_metrics["continuous/top_threshold"]
                for entry in cached:
                    entry["continuous_score"] = entry["selected"].float()
            else:
                severity_scale = max(
                    threshold_from_histogram(
                        severity_hist,
                        cfg.continuous_quantile,
                        cfg.quantile_max,
                        cfg.quantile_min,
                    ),
                    cfg.quantile_min,
                )
                for entry in cached:
                    entry["continuous_score"] = continuous_scores_from_severity(
                        entry["raw_severity"],
                        entry["selected"],
                        severity_scale,
                        cfg.continuous_floor,
                    )

            shuffle_change = torch.zeros(2, dtype=torch.float64, device=device)
            if cfg.continuous_shuffle:
                generator = torch.Generator(device=device)
                rank = torch.distributed.get_rank() if distributed else 0
                generator.manual_seed(
                    cfg.random_seed + 97_409 + int(data.meta_info.get("global_steps", 0)) + rank * 1_000_003
                )
                original_scores = [entry["continuous_score"] for entry in cached]
                shuffled_scores = stratified_shuffle_scores(
                    [entry["advantages"] for entry in cached],
                    [entry["mask"] for entry in cached],
                    [entry["selected"] for entry in cached],
                    original_scores,
                    cfg.stratified_bins,
                    cfg.quantile_min,
                    cfg.quantile_max,
                    generator,
                )
                for entry, original, shuffled in zip(cached, original_scores, shuffled_scores, strict=True):
                    chosen = entry["selected"].bool()
                    shuffle_change[0] += (original[chosen] - shuffled[chosen]).abs().sum().double()
                    shuffle_change[1] += chosen.sum().double()
                    entry["continuous_score"] = shuffled
                if distributed:
                    torch.distributed.all_reduce(shuffle_change)
                continuous_metrics["continuous/shuffle_mean_abs_change"] = float(
                    shuffle_change[0] / shuffle_change[1].clamp(min=1)
                )

            def global_extra(candidate):
                value = torch.zeros((), dtype=torch.float64, device=device)
                for entry in cached:
                    value += (candidate * entry["continuous_score"].double()).clamp(
                        max=cfg.w_max - 1
                    ).sum()
                if distributed:
                    torch.distributed.all_reduce(value)
                return float(value)

            # Preserve both original kappa's maximum intervention and the
            # global mass-growth budget. Bisection accounts for per-token caps.
            capacity = global_extra(lam)
            target = min(cfg.mass_growth_budget * valid_count, capacity)
            if reference_target is not None:
                target = min(target, reference_target)
            lo, hi = 0.0, lam
            for _ in range(32):
                mid = (lo + hi) / 2
                if global_extra(mid) < target:
                    lo = mid
                else:
                    hi = mid
            lam = (lo + hi) / 2
            score_sum = torch.zeros((), dtype=torch.float64, device=device)
            for entry in cached:
                score_sum += entry["continuous_score"].double().sum()
            if distributed:
                torch.distributed.all_reduce(score_sum)
            continuous_metrics = {
                **continuous_metrics,
                "continuous/severity_scale_q": severity_scale,
                "continuous/score_mean_selected": float(score_sum) / max(selected_count, 1),
                "continuous/target_mass_growth": target / valid_count,
                "continuous/score_mode_id": {
                    "delta": 0.0,
                    "surrogate": 1.0,
                    "rank": 2.0,
                    "topq": 3.0,
                }[cfg.continuous_score_mode],
                "continuous/shuffle_enabled": 1.0 if cfg.continuous_shuffle else 0.0,
            }

        # ---- pass 4: the single real update ----
        started = time.monotonic()
        self.actor_optimizer.zero_grad()
        weighted_loss = 0.0
        diagnostics = torch.zeros(len(DIAG_KEYS), dtype=torch.float64, device=device)
        mass = torch.zeros(2, dtype=torch.float64, device=device)
        for micro_batch, entry in zip(micro_batches, cached, strict=True):
            if cfg.weight_mode == "budget":
                weights = token_weights_budget(entry["selected"], entry["mask"], lam, cfg.w_max)
            elif cfg.weight_mode == "continuous_budget":
                weights = token_weights_continuous_budget(
                    entry["continuous_score"], entry["mask"], lam, cfg.w_max
                )
            else:
                weights = token_weights(
                    entry["selected"], entry["mask"], lam, cfg.w_max, valid_count, selected_count
                )
            weights = attenuate_opposite_weights(
                weights, entry["opposite_selected"], entry["mask"], cfg.opposite_weight_scale
            )
            if audit_enabled:
                entry["weights"] = weights.detach()
            diagnostics += selection_diagnostics(entry["selected"], entry["mask"], entry["advantages"])
            mass[0] += weights.sum().double()
            mass[1] += entry["mask"].sum().double()
            _, log_probs, *_ = self._forward_micro_batch(dict(micro_batch.batch), temperature=temperature)
            total = weighted_policy_sum(log_probs, entry["advantages"], weights, entry["mask"])
            loss = -total / valid_count * world_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite disturbed-OPD loss; refusing optimizer update")
            loss.backward()
            weighted_loss += float(loss.detach())
        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()
        timings["real_update"] = time.monotonic() - started
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite disturbed-OPD gradient; optimizer step was skipped")

        if audit_enabled:
            # Evaluate the exact same sampled trajectories after the real AdamW
            # update.  This directly measures functional damage/rescue without
            # storing parameter-sized gradients.
            post_started = time.monotonic()
            with torch.no_grad():
                for micro_batch, entry in zip(micro_batches, cached, strict=True):
                    post_inputs = dict(micro_batch.batch)
                    teacher_ids = post_inputs.get("teacher_top_k_ids") if audit_kl_topk else None
                    requested_topk = audit_kl_topk if audit_kl_topk else cfg.audit_topk
                    post_entropy, post_log_probs, post_topk_ids, post_topk_log_probs = self._forward_micro_batch(
                        post_inputs,
                        temperature=temperature,
                        calculate_entropy=cfg.audit_post_update,
                        top_k=requested_topk,
                        student_top_k_ids=teacher_ids,
                    )
                    entry["log_probs_after"] = post_log_probs.detach()
                    entry["entropy_after"] = post_entropy.detach() if post_entropy is not None else None
                    entry["topk_ids_after"] = (
                        post_topk_ids.detach()
                        if post_topk_ids is not None and not audit_kl_topk
                        else None
                    )
                    entry["topk_log_probs_after"] = (
                        post_topk_log_probs.detach()
                        if post_topk_log_probs is not None and not audit_kl_topk
                        else None
                    )
                    entry["distribution_kl_after"] = (
                        topk_tail_forward_kl(
                            post_inputs["teacher_top_k_log_probs"], post_topk_log_probs
                        )
                        if audit_kl_topk
                        else None
                    )
            timings["audit_post_update_forward"] = time.monotonic() - post_started

            # Per-step relation audit.  It is deliberately computed from the
            # true probe-defined up/down masks even for random selectors, so
            # selector ablations remain directly comparable.
            entropy_hist = torch.zeros(cfg.quantile_bins + 1, dtype=torch.int64, device=device)
            kl_hist = torch.zeros_like(entropy_hist)
            for entry in cached:
                entropy_hist += abs_histogram(
                    entry["entropy"], entry["mask"], cfg.quantile_bins, cfg.quantile_max, cfg.quantile_min
                )
                if audit_kl_topk:
                    kl_hist += abs_histogram(
                        entry["distribution_kl_before"],
                        entry["mask"],
                        cfg.quantile_bins,
                        cfg.quantile_max,
                        cfg.quantile_min,
                    )
            if distributed:
                torch.distributed.all_reduce(entropy_hist)
                torch.distributed.all_reduce(kl_hist)
            entropy_q25 = threshold_from_histogram(
                entropy_hist, 0.25, cfg.quantile_max, cfg.quantile_min
            )
            entropy_q75 = threshold_from_histogram(
                entropy_hist, 0.75, cfg.quantile_max, cfg.quantile_min
            )
            kl_q25 = (
                threshold_from_histogram(kl_hist, 0.25, cfg.quantile_max, cfg.quantile_min)
                if audit_kl_topk
                else 0.0
            )
            kl_q75 = (
                threshold_from_histogram(kl_hist, 0.75, cfg.quantile_max, cfg.quantile_min)
                if audit_kl_topk
                else 0.0
            )

            group_names = ("up", "down", "eos", "last")
            # Global totals: N, sum(H), sum(H^2), sum(KL), sum(KL^2),
            # high/low-H counts, high/low-KL counts.
            relation_totals = torch.zeros(9, dtype=torch.float64, device=device)
            # Per group: count, sum(H), sum(KL), high/low-H, high/low-KL.
            relation_groups = torch.zeros((len(group_names), 7), dtype=torch.float64, device=device)
            for micro_batch, entry in zip(micro_batches, cached, strict=True):
                inputs = dict(micro_batch.batch)
                valid = entry["mask"].bool()
                entropy_value = entry["entropy"].float()
                kl_value = (
                    entry["distribution_kl_before"].float()
                    if audit_kl_topk
                    else torch.zeros_like(entropy_value)
                )
                lengths = valid.sum(-1)
                positions = torch.arange(valid.shape[1], device=device).expand_as(valid)
                last = valid & (positions == (lengths - 1).clamp_min(0).unsqueeze(-1))
                eos = valid & (inputs["responses"] == cfg.audit_eos_token_id)
                entry["audit_is_eos"] = eos
                entry["audit_is_last"] = last
                high_entropy = valid & (entropy_value >= entropy_q75)
                low_entropy = valid & (entropy_value <= entropy_q25)
                high_kl = valid & (kl_value >= kl_q75) if audit_kl_topk else torch.zeros_like(valid)
                low_kl = valid & (kl_value <= kl_q25) if audit_kl_topk else torch.zeros_like(valid)

                h = entropy_value[valid].double()
                k = kl_value[valid].double()
                relation_totals += torch.stack(
                    (
                        torch.tensor(float(h.numel()), dtype=torch.float64, device=device),
                        h.sum(),
                        (h * h).sum(),
                        k.sum(),
                        (k * k).sum(),
                        high_entropy.sum().double(),
                        low_entropy.sum().double(),
                        high_kl.sum().double(),
                        low_kl.sum().double(),
                    )
                )
                groups = (entry["audit_up"], entry["audit_down"], eos, last)
                for index, group_mask in enumerate(groups):
                    chosen = valid & group_mask.bool()
                    relation_groups[index] += torch.stack(
                        (
                            chosen.sum().double(),
                            entropy_value[chosen].double().sum(),
                            kl_value[chosen].double().sum(),
                            (chosen & high_entropy).sum().double(),
                            (chosen & low_entropy).sum().double(),
                            (chosen & high_kl).sum().double(),
                            (chosen & low_kl).sum().double(),
                        )
                    )
            if distributed:
                torch.distributed.all_reduce(relation_totals)
                torch.distributed.all_reduce(relation_groups)

            n = float(relation_totals[0])
            mean_h = float(relation_totals[1] / max(n, 1.0))
            mean_k = float(relation_totals[3] / max(n, 1.0))
            var_h = max(float(relation_totals[2] / max(n, 1.0)) - mean_h**2, 0.0)
            var_k = max(float(relation_totals[4] / max(n, 1.0)) - mean_k**2, 0.0)

            def point_biserial(count, selected_sum, total_mean, total_var):
                probability = count / max(n, 1.0)
                variance_binary = probability * (1.0 - probability)
                if variance_binary <= 0 or total_var <= 0:
                    return 0.0
                covariance = selected_sum / max(n, 1.0) - probability * total_mean
                return covariance / math.sqrt(variance_binary * total_var)

            audit_metrics = {
                "audit/distribution_kl_topk": float(audit_kl_topk),
                "audit/eos_token_id": float(cfg.audit_eos_token_id),
                "audit/entropy_mean": mean_h,
                "audit/entropy_q25": entropy_q25,
                "audit/entropy_q75": entropy_q75,
                "audit/kl_mean": mean_k,
                "audit/kl_q25": kl_q25,
                "audit/kl_q75": kl_q75,
            }
            total_high_h, total_low_h = float(relation_totals[5]), float(relation_totals[6])
            total_high_k, total_low_k = float(relation_totals[7]), float(relation_totals[8])
            for index, name in enumerate(group_names):
                count, sum_h, sum_k, high_h, low_h, high_k, low_k = map(float, relation_groups[index])
                audit_metrics.update(
                    {
                        f"audit/{name}_count": count,
                        f"audit/{name}_fraction": count / max(n, 1.0),
                        f"audit/{name}_entropy_mean": sum_h / max(count, 1.0),
                        f"audit/{name}_kl_mean": sum_k / max(count, 1.0),
                        f"audit/{name}_high_entropy_share": high_h / max(count, 1.0),
                        f"audit/{name}_low_entropy_share": low_h / max(count, 1.0),
                        f"audit/{name}_rate_among_high_entropy": high_h / max(total_high_h, 1.0),
                        f"audit/{name}_rate_among_low_entropy": low_h / max(total_low_h, 1.0),
                        f"audit/{name}_high_kl_share": high_k / max(count, 1.0),
                        f"audit/{name}_low_kl_share": low_k / max(count, 1.0),
                        f"audit/{name}_rate_among_high_kl": high_k / max(total_high_k, 1.0),
                        f"audit/{name}_rate_among_low_kl": low_k / max(total_low_k, 1.0),
                        f"audit/{name}_entropy_point_biserial": point_biserial(
                            count, sum_h, mean_h, var_h
                        ),
                        f"audit/{name}_kl_point_biserial": point_biserial(
                            count, sum_k, mean_k, var_k
                        ),
                    }
                )
            up_count = float(relation_groups[0, 0])
            down_count = float(relation_groups[1, 0])
            eos_count = float(relation_groups[2, 0])
            last_count = float(relation_groups[3, 0])
            terminal_overlap = torch.zeros(4, dtype=torch.float64, device=device)
            for entry in cached:
                terminal_overlap[0] += (entry["audit_is_eos"] & entry["audit_up"]).sum().double()
                terminal_overlap[1] += (entry["audit_is_eos"] & entry["audit_down"]).sum().double()
                terminal_overlap[2] += (entry["audit_is_last"] & entry["audit_up"]).sum().double()
                terminal_overlap[3] += (entry["audit_is_last"] & entry["audit_down"]).sum().double()
            if distributed:
                torch.distributed.all_reduce(terminal_overlap)
            eos_up, eos_down, last_up, last_down = map(float, terminal_overlap)
            audit_metrics.update(
                {
                    "audit/eos_up_fraction": eos_up / max(eos_count, 1.0),
                    "audit/eos_down_fraction": eos_down / max(eos_count, 1.0),
                    "audit/up_eos_share": eos_up / max(up_count, 1.0),
                    "audit/down_eos_share": eos_down / max(down_count, 1.0),
                    "audit/last_up_fraction": last_up / max(last_count, 1.0),
                    "audit/last_down_fraction": last_down / max(last_count, 1.0),
                    "audit/up_last_share": last_up / max(up_count, 1.0),
                    "audit/down_last_share": last_down / max(down_count, 1.0),
                }
            )

            # Every DP rank owns a disjoint response shard, so each writes an
            # atomic per-step parquet shard.  The driver-side rollout JSONL uses
            # the same audit_response_id for an exact join back to prompt/text.
            from pathlib import Path

            import pyarrow as pa
            import pyarrow.parquet as pq

            rank = torch.distributed.get_rank() if distributed else 0
            step = int(data.meta_info.get("global_steps", 0))
            step_dir = Path(cfg.audit_dir) / f"step_{step:05d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            columns = {
                "global_step": [],
                "rank": [],
                "audit_response_id": [],
                "token_index": [],
                "relative_position": [],
                "response_length": [],
                "token_id": [],
                "rollout_old_logp": [],
                "actor_logp_before": [],
                "teacher_logp": [],
                "advantage": [],
                "entropy_before": [],
                "probe_logp": [],
                "delta_probe": [],
                "probe_damage": [],
                "mask_up": [],
                "mask_down": [],
                "probe_selected": [],
                "intervened": [],
                "token_weight": [],
                "actor_logp_after": [],
                "delta_real": [],
                "real_damage": [],
                "entropy_after": [],
                "teacher_entropy": [],
                "distribution_kl_before": [],
                "distribution_kl_after": [],
                "is_eos": [],
                "is_last_token": [],
                "probe_real_sign_agree": [],
                "topk_ids_before": [],
                "topk_log_probs_before": [],
                "topk_ids_after": [],
                "topk_log_probs_after": [],
            }
            for tag in diagnostic_tags:
                columns[f"diag_{tag}_delta"] = []
                columns[f"diag_{tag}_up"] = []
                columns[f"diag_{tag}_down"] = []

            def flat_valid(tensor, valid):
                return tensor[valid].detach().cpu()

            for micro_batch, entry in zip(micro_batches, cached, strict=True):
                inputs = dict(micro_batch.batch)
                valid = entry["mask"].bool()
                batch_size, response_width = valid.shape
                lengths = valid.sum(-1)
                positions = torch.arange(response_width, device=valid.device).expand(batch_size, -1)
                relative = positions.float() / lengths.clamp(min=1).unsqueeze(-1).float()
                response_ids = inputs.get("audit_response_id")
                if response_ids is None:
                    response_ids = torch.arange(batch_size, device=valid.device, dtype=torch.int64)
                response_ids = response_ids.reshape(-1, 1).expand(-1, response_width)
                expanded_lengths = lengths.reshape(-1, 1).expand(-1, response_width)
                delta_real = (entry["log_probs_after"] - entry["log_probs"]).float()
                teacher_logp = inputs["old_log_probs"].float() + entry["advantages"].float()
                values = {
                    "global_step": torch.full_like(positions, step, dtype=torch.int64),
                    "rank": torch.full_like(positions, rank, dtype=torch.int64),
                    "audit_response_id": response_ids,
                    "token_index": positions,
                    "relative_position": relative,
                    "response_length": expanded_lengths,
                    "token_id": inputs["responses"],
                    "rollout_old_logp": inputs["old_log_probs"],
                    "actor_logp_before": entry["log_probs"],
                    "teacher_logp": teacher_logp,
                    "advantage": entry["advantages"],
                    "entropy_before": entry["entropy"],
                    "probe_logp": entry["log_probs"].float() + entry["change"],
                    "delta_probe": entry["change"],
                    "probe_damage": -entry["advantages"].float() * entry["change"],
                    "mask_up": entry["audit_up"],
                    "mask_down": entry["audit_down"],
                    "probe_selected": entry["selected"],
                    "intervened": (entry["weights"].float() - 1.0).abs() > 1e-8,
                    "token_weight": entry["weights"],
                    "actor_logp_after": entry["log_probs_after"],
                    "delta_real": delta_real,
                    "real_damage": -entry["advantages"].float() * delta_real,
                    "entropy_after": entry["entropy_after"],
                    "teacher_entropy": entry["teacher_entropy"],
                    "distribution_kl_before": entry["distribution_kl_before"],
                    "distribution_kl_after": entry["distribution_kl_after"],
                    "is_eos": entry["audit_is_eos"],
                    "is_last_token": entry["audit_is_last"],
                    "probe_real_sign_agree": entry["change"].sign() == delta_real.sign(),
                }
                for name, tensor in values.items():
                    if tensor is None:
                        columns[name].extend([None] * int(valid.sum()))
                    else:
                        columns[name].extend(flat_valid(tensor, valid).tolist())
                for tag in diagnostic_tags:
                    columns[f"diag_{tag}_delta"].extend(
                        flat_valid(entry[f"diag/{tag}/delta"], valid).tolist()
                    )
                    columns[f"diag_{tag}_up"].extend(
                        flat_valid(entry[f"diag/{tag}/up"], valid).tolist()
                    )
                    columns[f"diag_{tag}_down"].extend(
                        flat_valid(entry[f"diag/{tag}/down"], valid).tolist()
                    )
                for name in (
                    "topk_ids_before",
                    "topk_log_probs_before",
                    "topk_ids_after",
                    "topk_log_probs_after",
                ):
                    tensor = entry[name]
                    if tensor is None:
                        columns[name].extend([None] * int(valid.sum()))
                    else:
                        columns[name].extend(tensor[valid].detach().cpu().tolist())

            table = pa.Table.from_pydict(columns)
            final_path = step_dir / f"rank_{rank:02d}.parquet"
            temp_path = step_dir / f"rank_{rank:02d}.parquet.tmp"
            pq.write_table(table, temp_path, compression="zstd", row_group_size=131072)
            os.replace(temp_path, final_path)

        if not audit_enabled:
            audit_metrics = {}
        if distributed:
            torch.distributed.all_reduce(diagnostics)
            torch.distributed.all_reduce(mass)
        diag = dict(zip(DIAG_KEYS, diagnostics.tolist(), strict=True))
        metrics = {
            "actor/grad_norm": float(grad_norm.detach()),
            "actor/pg_loss": weighted_loss,
            "disturbed/weight_mass_growth": float(mass[0] / mass[1].clamp(min=1)) - 1.0,
            "disturbed/opposite_weight_scale": cfg.opposite_weight_scale,
            "disturbed/opposite_selected_fraction": opposite_count / valid_count,
            "disturbed/opposite_removed_mass_fraction": (
                (1.0 - cfg.opposite_weight_scale) * opposite_count / valid_count
            ),
            "disturbed/lambda_uncapped": lam_uncapped,
            # V composition: without these the selected set is a single opaque
            # fraction and any account of *why* it helps or hurts is speculation.
            **{f"disturbed/vfrac_q{q}": (diag[f"sel_q{q}"] / diag[f"val_q{q}"]) if diag[f"val_q{q}"] else 0.0
               for q in range(4)},
            "disturbed/vfrac_adv_positive": (diag["sel_apos"] / diag["val_apos"]) if diag["val_apos"] else 0.0,
            "disturbed/vfrac_adv_negative": (diag["sel_aneg"] / diag["val_aneg"]) if diag["val_aneg"] else 0.0,
            "disturbed/selected_share_negative": (diag["sel_aneg"] / diag["selected"]) if diag["selected"] else 0.0,
            "disturbed/gain": gain,
            "disturbed/damage": damage,
            "disturbed/lambda": lam,
            "disturbed/lambda_max": lam_max if math.isfinite(lam_max) else -1.0,
            "disturbed/selected_fraction": selected_count / valid_count,
            "disturbed/gain_nonpositive": 1.0 if fell_back else 0.0,
            "disturbed/a_min_threshold": threshold,
            "disturbed/delta_param_changed_fraction": moved / max(counted, 1),
            "disturbed/probe_grad_norm": float(probe_norm.detach()),
            "disturbed/alpha": cfg.alpha,
            "disturbed/probe_momentum": 1.0 if cfg.probe_momentum else 0.0,
            "disturbed/optimizer_steps": 1.0,
            **continuous_metrics,
            **diagnostic_metrics,
            **audit_metrics,
        }
        for name, value in timings.items():
            metrics[f"timing_s/disturbed_{name}"] = value
        return metrics
