# phase_slip/sampler.py

"""
 Original Author: Michael Christian Morgan
 2025.12.03
 Github: https://github.com/Mmorgan-ML
 Twitter: @Mmorgan_ML
 Email: mmorgankorea@gmail.com
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import copy
import random

class PhaseSlipSampler:
    def __init__(self, model, tokenizer, 
                 stagnation_threshold: float = 0.6, 
                 patience: int = 5, 
                 noise_scale: float = 0.03, # Epsilon (Small)
                 perturbation_window: int = 12, # W (Effective Horizon)
                 noise_type: str = "ortho_rotation",
                 use_ngram_check: bool = True, # Kept for legacy compatibility
                 mask_prompt: bool = True,
                 shock_temperature_factor: float = 1.0,
                 shock_cooldown: int = 0, # Unused in Continuous Mode
                 logit_fusion_alpha: float = 0.45, # Alpha (Strong Anchor Default)
                 speculative_candidates: int = 0,
                 target_heads: List[Tuple[int, int]] = None,
                 blend_beta: float = 0.15, # Beta (Blend Strength)
                 rotation_mechanism: str = "vector", # Default: Optimized Manifold Projection
                 dynamic_alpha: bool = True, # Default: Enabled Confidence Gating
                 stochastic_skip_ratio: float = 0.0): # Default: 0.0 (Stability Mode)
        """
        Args:
            stagnation_threshold (float): Legacy.
            patience (int): Legacy.
            noise_scale (float): Magnitude of rotation (Epsilon).
            perturbation_window (int): Size of the active perturbation window (W).
            noise_type (str): 'ortho_rotation', 'gaussian'.
            use_ngram_check (bool): Legacy.
            mask_prompt (bool): Protect system prompt.
            shock_temperature_factor (float): Legacy.
            shock_cooldown (int): Legacy.
            logit_fusion_alpha (float): Strength of Clean Anchor (0.0 = Pure Noise, 1.0 = Pure Clean).
            speculative_candidates (int): Legacy.
            target_heads (List): Specific (Layer, Head) tuples to target.
            blend_beta (float): Soft blend factor (0.0 = Original, 1.0 = Full Rotation).
            rotation_mechanism (str): "matrix" for original QR approach, "vector" for Manifold Projection.
            dynamic_alpha (bool): If True, alpha increases when model is confident (Low Entropy).
            stochastic_skip_ratio (float): Probability to skip the Phantom pass entirely (Speed Optimization).
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # --- PHYSICS PARAMETERS ---
        self.noise_scale = noise_scale
        self.blend_beta = blend_beta
        self.logit_fusion_alpha = logit_fusion_alpha
        
        # --- CONFIGURATION ---
        self.perturbation_window = perturbation_window
        self.noise_type = noise_type
        self.mask_prompt = mask_prompt
        self.target_heads = target_heads if target_heads else []
        
        # --- OPTIMIZATIONS ---
        self.rotation_mechanism = rotation_mechanism
        self.dynamic_alpha = dynamic_alpha
        self.stochastic_skip_ratio = stochastic_skip_ratio
        
        # Legacy params kept for compatibility
        self.stagnation_threshold = stagnation_threshold
        self.patience_limit = patience
        self.use_ngram_check = use_ngram_check
        self.shock_temperature_factor = shock_temperature_factor

    def calibrate_heads(self, prompt_text: str = "The scientific method is a process that involves", search_layers: int = 6):
        """
        Scans layers to find heads that drive diversity without breaking the model.
        Uses gentle parameters (eps=0.03, beta=0.15) for calibration.
        """
        print(f"   [Calibration] Scanning last {search_layers} layers for creative heads...")
        device = self.model.device
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            base_out = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        base_text = self.tokenizer.decode(base_out[0])
        
        impact_scores = []
        n_layers = self.model.config.n_layer
        n_heads = self.model.config.n_head
        start_layer = max(0, n_layers - search_layers)
        
        for layer in range(start_layer, n_layers):
            for head in range(n_heads):
                # Use Gentle Sampler for Calibration
                test_sampler = PhaseSlipSampler(
                    self.model, self.tokenizer,
                    noise_type="ortho_rotation",
                    target_heads=[(layer, head)],
                    noise_scale=0.03, 
                    blend_beta=0.15,  
                    perturbation_window=5,
                    rotation_mechanism=self.rotation_mechanism # Use same mechanism as parent
                )
                
                try:
                    out_text = test_sampler.generate(prompt_text, max_new_tokens=10, verbose=False)
                    
                    # Safety Check: If output is empty or only whitespace, head is critical (Avoid).
                    if not out_text.strip():
                        continue

                    set_base = set(base_text.split())
                    set_out = set(out_text.split())
                    intersection = len(set_base.intersection(set_out))
                    union = len(set_base.union(set_out))
                    similarity = intersection / union if union > 0 else 1.0
                    
                    impact = 1.0 - similarity
                    impact_scores.append(((layer, head), impact))
                except Exception:
                    continue
        
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        top_heads = [x[0] for x in impact_scores[:3]]
        print(f"   [Calibration] Identified Top Heads: {top_heads}")
        
        self.target_heads = top_heads
        return top_heads

    def latent_perturbation(self, past_key_values, prompt_len: int = 0):
        """
        Applies Targeted Soft-Blend Perturbation.
        Returns a NEW tuple of tensors (Ephemeral).
        """
        if not isinstance(past_key_values, tuple): return past_key_values

        new_memory_list = []
        
        for layer_idx, layer in enumerate(past_key_values):
            key, value = layer
            device = key.device
            seq_len = key.shape[-2]
            head_dim = key.shape[-1]
            
            # --- 1. CALCULATE WINDOW MASK ---
            # Strictly rolling window: [End-W : End]
            mask = torch.zeros_like(key)
            start_idx = max(prompt_len, seq_len - self.perturbation_window)
            
            if start_idx < seq_len:
                mask[:, :, start_idx:, :] = 1.0
            else:
                # Window hasn't started yet or is protected
                new_memory_list.append((key, value))
                continue

            # --- 2. CALCULATE HEAD MASK ---
            # If targets exist, zero out non-targets
            if self.target_heads:
                head_mask_tensor = torch.zeros(key.shape[1], device=device)
                has_target = False
                for (l, h) in self.target_heads:
                    if l == layer_idx:
                        head_mask_tensor[h] = 1.0
                        has_target = True
                
                head_mask_tensor = head_mask_tensor.view(1, -1, 1, 1)
                
                if has_target:
                    mask = mask * head_mask_tensor
                else:
                    new_memory_list.append((key, value))
                    continue

            # --- 3. ORTHONORMAL ROTATION ---
            if self.noise_type == "ortho_rotation":
                
                if self.rotation_mechanism == "matrix":
                    # --- LEGACY METHOD: MATRIX QR DECOMPOSITION ---
                    # Generate random matrix M
                    M = torch.randn(head_dim, head_dim, device=device)
                    Q, R = torch.linalg.qr(M)
                    
                    # R = I + eps * (Q - I)
                    I = torch.eye(head_dim, device=device)
                    Rot = I + self.noise_scale * (Q - I)
                    
                    # Apply Rotation to Values: V @ Rot.T
                    rotated_v = torch.matmul(value, Rot.t())
                
                else:
                    # --- PRODUCTION METHOD: VECTOR MANIFOLD PROJECTION ---
                    # More efficient O(n) and preserves magnitude better
                    current_norm = torch.norm(value, dim=-1, keepdim=True)
                    
                    # Generate random target direction
                    target_noise = torch.randn_like(value)
                    
                    # Lerp towards noise to create "Rotated" vector
                    # Note: We do not add, we interpolate, then normalize.
                    # This simulates rotating the vector on the sphere surface.
                    rotated_v_raw = (1.0 - self.noise_scale) * value + (self.noise_scale * target_noise)
                    
                    # Restore Magnitude (Project back to Manifold)
                    rotated_v = rotated_v_raw * (current_norm / (torch.norm(rotated_v_raw, dim=-1, keepdim=True) + 1e-8))

                # --- 4. SOFT BLEND ---
                # V' = (1-beta)V + beta(RV)
                # Apply only where mask is 1
                blended_v = (1.0 - self.blend_beta) * value + (self.blend_beta * rotated_v)
                
                new_val = value * (1 - mask) + blended_v * mask
                new_key = key # Keys untouched
                
                new_memory_list.append((new_key, new_val))
                continue
            
            # Fallback for Gaussian (if needed for baselines)
            elif self.noise_type == "gaussian":
                noise = torch.randn_like(value) * (value.std() * self.noise_scale)
                new_val = value + (noise * mask)
                new_memory_list.append((key, new_val))
                continue

            new_memory_list.append((key, value))
            
        return tuple(new_memory_list)

    def generate(self, prompt_text: str, max_new_tokens: int = 50, verbose: bool = False, temperature: float = 0.65) -> str:
        """
        Generate text using Phase-Slip.
        Default temperature set to 0.65 for optimal coherence.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        prompt_len = input_ids.shape[-1]
        
        # Initial Forward Pass (Clean)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :] 

        generated_ids_list = input_ids[0].tolist()

        for i in range(max_new_tokens):
            with torch.no_grad():
                # --- CONTINUOUS STEERING (Every Step) ---
                
                # Check Stochastic Skip
                # If skip_ratio is > 0.0, we random skip the Phantom Pass.
                do_perturbation = True
                if self.stochastic_skip_ratio > 0.0:
                    if random.random() < self.stochastic_skip_ratio:
                        do_perturbation = False

                clean_logits = next_token_logits
                
                if do_perturbation:
                    # 1. PERTURB (Ephemeral Copy)
                    noisy_past = self.latent_perturbation(past_key_values, prompt_len=prompt_len)
                    
                    # 2. PHANTOM FORWARD PASS (To get Steering Logits)
                    # We recompute logits using the *last token* and the *noisy cache*
                    last_token_input = torch.tensor([[generated_ids_list[-1]]], device=device)
                    phantom_outputs = self.model(last_token_input, past_key_values=noisy_past, use_cache=True)
                    phantom_logits = phantom_outputs.logits[:, -1, :]
                    
                    # 3. LOGIT ANCHORING
                    # Calculate Effective Alpha
                    current_alpha = self.logit_fusion_alpha
                    
                    if self.dynamic_alpha:
                        # Calculate Entropy of Clean Logits
                        probs = F.softmax(clean_logits / temperature, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        
                        # Sigmoid gate centered around entropy 2.0
                        # confidence score 0.0 to 1.0
                        confidence = torch.sigmoid(2.5 - entropy).item() 
                        
                        # Interpolate between base alpha and 1.0
                        current_alpha = self.logit_fusion_alpha + (1.0 - self.logit_fusion_alpha) * confidence

                    # Final = (1-alpha) * Perturbed + alpha * Clean
                    sampling_logits = ((1 - current_alpha) * phantom_logits) + \
                                      (current_alpha * clean_logits)
                else:
                    # SKIP MODE: Pure Clean Forward Pass (Standard Sampling this step)
                    sampling_logits = clean_logits

                # 5. CONSTRAINED SAMPLING
                # Top-K / Top-P Filtering (Hardcoded defaults for stability)
                top_k = 40
                top_p = 0.92
                
                # Filter Logits
                probs = F.softmax(sampling_logits / temperature, dim=-1)
                
                # Apply Top-K
                if top_k > 0:
                    indices_to_remove = sampling_logits < torch.topk(sampling_logits, top_k)[0][..., -1, None]
                    sampling_logits[indices_to_remove] = -float('Inf')
                
                # Re-calc probs after filtering
                probs = F.softmax(sampling_logits / temperature, dim=-1)
                
                # Sample
                next_token_id = torch.multinomial(probs, num_samples=1)

                # --- UPDATE ---
                token_int = next_token_id.item()
                generated_ids_list.append(token_int)
                
                if token_int == self.tokenizer.eos_token_id:
                    break

                next_input = next_token_id.to(device)
                
                # 6. COMMIT TO CLEAN CACHE
                outputs = self.model(next_input, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        return self.tokenizer.decode(generated_ids_list, skip_special_tokens=True)