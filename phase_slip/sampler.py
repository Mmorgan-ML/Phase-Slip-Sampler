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
    def __init__(self, model, tokenizer, confusion_threshold: float = 4.0, cooldown: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.confusion_threshold = confusion_threshold
        self.last_flip_step = -10 
        self.cooldown_period = cooldown
        self.noise_scale = 0.02 

    def calculate_confusion(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def thermal_shock(self, past_key_values):
        """
        Applies Gaussian noise to the Key-Value cache.
        """
        # Detection: Handle both tuple (standard) and DynamicCache (newer HF)
        is_tuple = isinstance(past_key_values, tuple)
        
        # If it's a DynamicCache or other object, we might crash if we iterate blindly.
        # For GPT-2 standard, it is a tuple of tuples.
        if not is_tuple:
            # Fallback: If we can't easily modify it, return as is to prevent crash
            return past_key_values

        new_memory_list = []
        
        for layer in past_key_values:
            # Unpack layer (Key, Value)
            key, value = layer
            
            # Calculate noise relative to signal strength (Standard Deviation)
            noise_sigma_k = key.std() * self.noise_scale
            noise_sigma_v = value.std() * self.noise_scale

            # Create noise on the same device/dtype as the tensor
            key_noise = torch.randn_like(key) * noise_sigma_k
            val_noise = torch.randn_like(value) * noise_sigma_v
            
            # Add noise (Out of place addition to keep history clean)
            new_key = key + key_noise
            new_val = value + val_noise
            
            new_memory_list.append((new_key, new_val))
            
        return tuple(new_memory_list)

    def generate(self, prompt_text: str, max_new_tokens: int = 40, verbose: bool = False) -> str:
        # 1. Prepare Inputs
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        
        # Robust Device Detection
        device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        
        # 2. Initial Forward Pass
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        generated_ids = input_ids

        # 3. Generation Loop
        for i in range(max_new_tokens):
            with torch.no_grad(): # CRITICAL: Ensure no gradients are calculated
                confusion = self.calculate_confusion(next_token_logits)
                steps_since_flip = i - self.last_flip_step
                
                # --- PHASE SLIP LOGIC ---
                if confusion > self.confusion_threshold and steps_since_flip > self.cooldown_period:
                    if verbose:
                        print(f"   [Step {i}] Entropy Spike ({confusion:.2f}). Triggering Phase Slip...")
                    
                    past_key_values = self.thermal_shock(past_key_values)
                    
                    # High Temp Sampling
                    probs = F.softmax(next_token_logits / 1.5, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    self.last_flip_step = i 
                else:
                    # Greedy (Low Temp)
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                # ------------------------

                # Ensure next_token_id is on correct device
                next_token_id = next_token_id.to(device)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # Check EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Next Step Forward
                outputs = self.model(next_token_id, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)