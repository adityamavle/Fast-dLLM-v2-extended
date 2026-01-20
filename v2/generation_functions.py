from typing import Callable, Optional, Union
import torch
import types
from transformers.utils import auto_docstring, logging
from skip import TokenSkipPolicy, LayerSkipPolicy, SkipStats
from skip.model_hooks import apply_layer_skipping, register_hidden_state_hook, remove_hidden_state_hook

# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

MASK_COLOR = 0.5
TOKEN_COLOR = -0.5


@auto_docstring
class Fast_dLLM_QwenForCausalLM:

    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        # Skip parameters
        token_skip_enabled=False,
        token_tau=0.99,
        layer_skip_enabled=False,
        layer_tau=0.99,
        skip_stats=None,
    ):
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        # Initialize skip policies
        token_policy = TokenSkipPolicy(tau_token=token_tau, enabled=token_skip_enabled)
        layer_policy = LayerSkipPolicy(tau_layer=layer_tau, enabled=layer_skip_enabled)

        # Initialize stats if provided
        if skip_stats is not None:
            if hasattr(self, "model") and hasattr(self.model, "layers"):
                skip_stats.total_layers = len(self.model.layers)
            elif hasattr(self, "layers"):
                skip_stats.total_layers = len(self.layers)
            else:
                skip_stats.total_layers = 32  # fallback
            skip_stats.sequence_length = block_size
            skip_stats.token_tau = token_tau if token_skip_enabled else 0.0
            skip_stats.layer_tau = layer_tau if layer_skip_enabled else 0.0

        # Apply layer skipping hooks if enabled
        if layer_skip_enabled:
            apply_layer_skipping(self, layer_policy)
        
        # Register hidden state hook for token skipping (store as instance variable for cleanup)
        if token_skip_enabled:
            self._token_skip_hook = register_hidden_state_hook(self, token_policy)
        else:
            self._token_skip_hook = None

        if min_len > block_size:
            output = self.forward(
                input_ids=input_ids[:, : (min_len // block_size * block_size)],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break

            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, : (block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            block_past_key_values = None

            # Reset policies for new block
            token_policy.reset()
            layer_policy.reset()

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)

                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx] :] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1 :] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break

                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        # --- TOKEN-LEVEL SKIP (probe + stats variables must ALWAYS exist) ---
                        # default: full block window token count (B * block_size)
                        active_tokens = x_t[:, -block_size:].numel()
                        skip_mask = None
                        h_probe = None

                        if token_policy.enabled:
                            try:
                                h_probe = token_policy.compute_probe_hidden(
                                    self, x_t[:, -block_size:], past_key_values
                                )
                                skip_mask = token_policy.compute_skip_mask(
                                    h_probe, token_policy.h_prev
                                )
                                # number of tokens computed (not skipped) in probe mask
                                active_tokens = (~skip_mask).sum().item()
                            except Exception:
                                skip_mask = None
                                h_probe = None
                                active_tokens = x_t[:, -block_size:].numel()

                        # Reset layer policy for this forward so executed count is per-step
                        layer_policy.reset()

                        # --- forward path with token skipping ---
                        # If token skipping is enabled and we have a skip_mask, we need to handle it
                        # Note: We still process all tokens (attention needs full context), but we track
                        # skipped tokens for FLOPs calculation and can optionally replace their outputs
                        use_token_skip = token_policy.enabled and skip_mask is not None and skip_mask.any()
                        
                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                ).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        
                        # Apply token skipping: replace logits for skipped tokens with logits from previous hidden states
                        # CRITICAL: Only skip tokens that are NOT masked (already unmasked/decoded)
                        # Masked tokens MUST be computed for denoising to work correctly
                        if use_token_skip and token_policy.h_prev_final is not None and skip_mask is not None:
                            # Align skip_mask with the current logits slice (start:end from full block)
                            if end is not None:
                                skip_mask_slice = skip_mask[:, start:end]
                                mask_idx_slice = mask_idx[:, start:end]
                            else:
                                skip_mask_slice = skip_mask[:, start:]
                                mask_idx_slice = mask_idx[:, start:]
                            
                            # ONLY skip tokens that are:
                            # 1. Marked for skipping (high similarity with previous step)
                            # 2. NOT masked (already unmasked/decoded in previous steps)
                            # Masked tokens MUST be computed for denoising to work
                            skip_mask_slice = skip_mask_slice & (~mask_idx_slice)
                            
                            if skip_mask_slice.any():
                                logits = token_policy.apply_token_skipping_to_logits(self, logits, skip_mask_slice)

                        # Update token policy cache: store probe for next step comparison
                        # Note: h_prev_final is automatically captured by the forward hook
                        if token_policy.enabled and (skip_mask is not None) and (h_probe is not None):
                            token_policy.h_prev = h_probe.clone()

                        # Record stats (now safe: active_tokens always defined)
                        if skip_stats is not None:
                            executed_layers = layer_policy.get_executed_count()
                            if executed_layers == 0:
                                executed_layers = skip_stats.total_layers
                            skip_stats.record_step(active_tokens, executed_layers)

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                        finished_flag = finished_flag | finished_row_flags

            # writeback block result
            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ]

            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

        # add not finished samples since max_new_tokens is reached
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        
        # Clean up: remove hidden state hook if it was registered
        if hasattr(self, '_token_skip_hook') and self._token_skip_hook is not None:
            remove_hidden_state_hook(self._token_skip_hook)
            self._token_skip_hook = None
        
        return finished_samples

    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024,
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
        # Skip parameters
        token_skip_enabled=False,
        token_tau=0.99,
        layer_skip_enabled=False,
        layer_tau=0.99,
        skip_stats=None,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        # Initialize skip policies
        token_policy = TokenSkipPolicy(tau_token=token_tau, enabled=token_skip_enabled)
        layer_policy = LayerSkipPolicy(tau_layer=layer_tau, enabled=layer_skip_enabled)

        # Initialize stats if provided
        if skip_stats is not None:
            if hasattr(self, "model") and hasattr(self.model, "layers"):
                skip_stats.total_layers = len(self.model.layers)
            elif hasattr(self, "layers"):
                skip_stats.total_layers = len(self.layers)
            else:
                skip_stats.total_layers = 32
            skip_stats.sequence_length = block_size
            skip_stats.token_tau = token_tau if token_skip_enabled else 0.0
            skip_stats.layer_tau = layer_tau if layer_skip_enabled else 0.0

        # Apply layer skipping hooks if enabled
        if layer_skip_enabled:
            apply_layer_skipping(self, layer_policy)
        
        # Register hidden state hook for token skipping
        if token_skip_enabled:
            self._token_skip_hook_viz = register_hidden_state_hook(self, token_policy)
        else:
            self._token_skip_hook_viz = None

        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        initial_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(
                input_ids=input_ids[:, : (input_ids.shape[1] // block_size * block_size)],
                use_cache=True,
                update_past_key_values=True,
            )
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for _ in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for _ in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state

            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - prompt_length % block_size),
                device=self.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            step = 0

            token_policy.reset()
            layer_policy.reset()

            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length : prompt_length + stop_token_idx] == mask_id).sum() == 0:
                        break

                mask_idx = (x_t[:, -block_size:] == mask_id)

                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length : prompt_length + stop_token_idx] == mask_id).sum() == 0:
                                break

                        # TOKEN-LEVEL SKIPPING: probe + safe defaults (use full block window token count)
                        active_tokens = x_t[:, -block_size:].numel()
                        skip_mask = None
                        h_probe = None
                        if token_policy.enabled:
                            try:
                                h_probe = token_policy.compute_probe_hidden(self, x_t[:, -block_size:], past_key_values)
                                skip_mask = token_policy.compute_skip_mask(h_probe, token_policy.h_prev)
                                active_tokens = (~skip_mask).sum().item()
                            except Exception:
                                skip_mask = None
                                h_probe = None
                                active_tokens = x_t[:, -block_size:].numel()

                        layer_policy.reset()

                        # Forward path with token skipping
                        use_token_skip = token_policy.enabled and skip_mask is not None and skip_mask.any()
                        
                        logits = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        ).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]
                        
                        # Apply token skipping: replace logits for skipped tokens
                        if use_token_skip and token_policy.h_prev_final is not None:
                            logits = token_policy.apply_token_skipping_to_logits(self, logits, skip_mask)

                        # Update token policy cache
                        if token_policy.enabled and (skip_mask is not None) and (h_probe is not None):
                            token_policy.h_prev = h_probe.clone()

                        if skip_stats is not None:
                            executed_layers = layer_policy.get_executed_count()
                            if executed_layers == 0:
                                executed_layers = skip_stats.total_layers
                            skip_stats.record_step(active_tokens, executed_layers)

                        step += 1
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, small_block_start_idx:small_block_end_idx], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]
                        for token_id in generated_tokens:
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                                current_state.append((token_text, TOKEN_COLOR))
                        yield current_state

            input_ids = x_t

        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, : stop_token_idx + original_input_length + 1]

        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            final_state.append((token_text, TOKEN_COLOR))

        yield final_state

        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text
        
        # Clean up: remove hidden state hook if it was registered
        if hasattr(self, '_token_skip_hook_viz') and self._token_skip_hook_viz is not None:
            remove_hidden_state_hook(self._token_skip_hook_viz)
            self._token_skip_hook_viz = None


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    model.mdm_sample_with_visualization = types.MethodType(
        Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model
    )
    return model
