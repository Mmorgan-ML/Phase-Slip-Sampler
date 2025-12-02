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
from typing import Tuple, Optional, Union

class PhaseSlipSampler:
    """
    A dynamic inference sampler that applies thermodynamic perturbations (Phase Slips)
    to the Key-Value cache of an LLM when entropy spikes.
    """

    def __init__(self, model, tokenizer, confusion_threshold: float = 4.0, cooldown: int = 3):
        """
        Initialize the sampler.

        Args:
            model: The HuggingFace model instance.
            tokenizer: The HuggingFace tokenizer.
            confusion_threshold (float): The entropy level required to trigger a shock.
            cooldown (int): Minimum steps between shocks to prevent destabilization.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.confusion_threshold = confusion_threshold
        self.last_flip_step = -10 
        self.cooldown_period = cooldown

    def calculate_confusion(self, logits: torch.Tensor) -> float:
        """
        Calculates the Shannon Entropy of the current token prediction distribution.
        Higher Entropy = Higher Confusion/Uncertainty.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def flip_memory(self, past_key_values: Tuple) -> Tuple:
        """
        Applies a 'Thermal Shock' (Gaussian Noise) to the Key-Value cache.
        
        This mimics simulated annealing, forcing the model to exit a local minimum
        (repetitive loop) by vibrating the semantic representation of the context.
        """
        # 0.03 is the empirically derived 'Goldilocks' noise level for GPT-2
        heat_level = 0.03 
        new_memory = ()
        
        for layer in past_key_values:
            key_tensor = layer[0]
            val_tensor = layer[1]
            
            # Generate random thermal noise
            key_heat = torch.randn_like(key_tensor) * heat_level
            val_heat = torch.randn_like(val_tensor) * heat_level
            
            # Inject noise into memory
            new_memory += ((key_tensor + key_heat, val_tensor + val_heat),)
        
        return new_memory

    def generate(self, prompt_text: str, max_new_tokens: int = 40) -> str:
        """
        Generates text using the Phase-Slip mechanism.
        
        Returns:
            str: The decoded generated text.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids
        
        # Initial forward pass to populate KV cache
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        generated_ids = input_ids

        for i in range(max_new_tokens):
            confusion = self.calculate_confusion(next_token_logits)
            steps_since_flip = i - self.last_flip_step
            
            # --- PHASE SLIP LOGIC ---
            if confusion > self.confusion_threshold and steps_since_flip > self.cooldown_period:
                # 1. Shock the Memory (Thermodynamic Injection)
                past_key_values = self.flip_memory(past_key_values)
                
                # 2. Spike the Temperature (Stochastic Sampling)
                # We divide logits by 1.5 to flatten the distribution (High Temp)
                probs = F.softmax(next_token_logits / 1.5, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                self.last_flip_step = i 
            else:
                # Standard Deterministic Behavior (Greedy)
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            # ------------------------

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            with torch.no_grad():
                outputs = self.model(next_token_id, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)