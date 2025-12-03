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

class PhaseSlipSampler:
    def __init__(self, model, tokenizer, stagnation_threshold: float = 0.6, patience: int = 5, noise_scale: float = 0.1):
        """
        Args:
            stagnation_threshold (float): If entropy drops below this, the model is 'too confident'.
            patience (int): How many low-entropy tokens to tolerate before triggering a shock.
            noise_scale (float): Magnitude of Gaussian noise injected into the KV cache.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # --- PHYSICS PARAMETERS ---
        self.stagnation_threshold = stagnation_threshold 
        self.patience_limit = patience
        self.current_patience = 0
        
        # Scaling factor for the "Thermal Shock"
        self.noise_scale = noise_scale

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Calculates Shannon Entropy.
        Low Entropy = High Confidence (or stuck in a loop).
        High Entropy = High Uncertainty (confusion).
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def latent_perturbation(self, past_key_values):
        """
        Injects non-destructive Gaussian noise into the Key-Value cache.
        This shifts the 'viewpoint' of the model's memory, forcing a re-evaluation of context.
        """
        is_tuple = isinstance(past_key_values, tuple)
        if not is_tuple: return past_key_values

        new_memory_list = []
        for layer in past_key_values:
            key, value = layer
            
            # We scale noise by the tensor's own standard deviation to keep it relative.
            # This ensures we don't destroy the memory, just "blur" it.
            noise_sigma_k = key.std() * self.noise_scale
            noise_sigma_v = value.std() * self.noise_scale

            key_noise = torch.randn_like(key) * noise_sigma_k
            val_noise = torch.randn_like(value) * noise_sigma_v
            
            new_key = key + key_noise
            new_val = value + val_noise
            
            new_memory_list.append((new_key, new_val))
            
        return tuple(new_memory_list)

    def generate(self, prompt_text: str, max_new_tokens: int = 50, verbose: bool = False, temperature: float = 1.0) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        
        # Pre-compute initial cache
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        generated_ids = input_ids

        for i in range(max_new_tokens):
            with torch.no_grad():
                current_entropy = self.calculate_entropy(next_token_logits)
                
                # --- LOGIC: DETECT STAGNATION ---
                # If entropy is dangerously low, we suspect a loop.
                if current_entropy < self.stagnation_threshold:
                    self.current_patience += 1
                else:
                    self.current_patience = 0 # Reset if the model shows creativity/uncertainty
                
                # --- TRIGGER: PHASE SLIP ---
                triggered = False
                if self.current_patience > self.patience_limit:
                    if verbose:
                        print(f"   [Step {i}] Stagnation Detected (Entropy: {current_entropy:.2f}). TRIGGERING SHOCK.")
                    
                    # 1. The Cool Hack: Shake the Memory
                    past_key_values = self.latent_perturbation(past_key_values)
                    
                    # 2. The Temperature Spike: Force a jump
                    # We temporarily double the temperature to ensure we pick a new path
                    current_temp = temperature * 2.0
                    
                    self.current_patience = 0 # Reset
                    triggered = True
                else:
                    current_temp = temperature

                # --- SAMPLING ---
                probs = F.softmax(next_token_logits / current_temp, dim=-1)
                
                # Use multinomial sampling to allow the shock to take effect
                next_token_id = torch.multinomial(probs, num_samples=1)

                next_token_id = next_token_id.to(device)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                outputs = self.model(next_token_id, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)