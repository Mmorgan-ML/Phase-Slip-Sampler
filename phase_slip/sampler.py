# phase_slip/sampler.py

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

class PhaseSlipSampler:
    def __init__(self, model, tokenizer, confusion_threshold: float = 3.5, cooldown: int = 4, max_interval: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        
        # --- THE PHYSICS PARAMETERS ---
        self.confusion_threshold = confusion_threshold # The "Ceiling" (Trigger Chaos)
        self.fact_shield = 0.05                        # The "Floor" (Protect Truth)
        
        self.last_flip_step = -10 
        self.cooldown_period = cooldown
        self.max_interval = max_interval
        self.noise_scale = 0.03 

    def calculate_confusion(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def thermal_shock(self, past_key_values):
        is_tuple = isinstance(past_key_values, tuple)
        if not is_tuple: return past_key_values

        new_memory_list = []
        for layer in past_key_values:
            key, value = layer
            
            # Scale noise by local variance
            noise_sigma_k = key.std() * self.noise_scale
            noise_sigma_v = value.std() * self.noise_scale

            key_noise = torch.randn_like(key) * noise_sigma_k
            val_noise = torch.randn_like(value) * noise_sigma_v
            
            new_key = key + key_noise
            new_val = value + val_noise
            
            new_memory_list.append((new_key, new_val))
            
        return tuple(new_memory_list)

    def generate(self, prompt_text: str, max_new_tokens: int = 40, verbose: bool = False) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        generated_ids = input_ids

        for i in range(max_new_tokens):
            with torch.no_grad():
                confusion = self.calculate_confusion(next_token_logits)
                steps_since_flip = i - self.last_flip_step
                
                # --- INTELLIGENT PHASE SLIP ---
                
                # 1. The Fact Shield: If confusion is effectively zero, we are reciting a fact.
                #    Do NOT slip, regardless of timers.
                is_fact = confusion < self.fact_shield
                
                # 2. Triggers
                natural_trigger = confusion > self.confusion_threshold
                boredom_trigger = steps_since_flip > self.max_interval
                
                # 3. Decision Matrix
                if not is_fact and (natural_trigger or boredom_trigger) and steps_since_flip > self.cooldown_period:
                    
                    if verbose:
                        reason = "Entropy Spike" if natural_trigger else "Boredom Check"
                        print(f"   [Step {i}] {reason} ({confusion:.2f}). Triggering Phase Slip...")
                    
                    past_key_values = self.thermal_shock(past_key_values)
                    
                    # Spike Temp
                    probs = F.softmax(next_token_logits / 1.5, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    self.last_flip_step = i 
                else:
                    # Standard Greedy (High Precision)
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                next_token_id = next_token_id.to(device)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                outputs = self.model(next_token_id, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)