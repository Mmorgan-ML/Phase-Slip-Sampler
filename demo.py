# demo.py

"""
 Original Author: Michael Christian Morgan
 2025.12.03
 Github: https://github.com/Mmorgan-ML
 Twitter: @Mmorgan_ML
 Email: mmorgankorea@gmail.com
"""

import torch
import traceback
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

# Fix: Silence huggingface warnings about padding
transformers.logging.set_verbosity_error()

def run_comparison():
    print("--- PHASE SLIP: VISUAL DEMO ---")
    print("Comparing: Greedy vs Standard vs Phase-Slip (Strong Anchor)")
    
    # 1. Load the Brain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 on {device}...")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Fix for GPT-2 Padding Issue
    model.config.pad_token_id = model.config.eos_token_id
    
    # 2. The Narrative Prompt
    # We use a story prompt to let the "Muse" shine without breaking factual logic
    prompt = "The ancient door creaked open, revealing a room filled with"
    print(f"\nPROMPT: '{prompt}'\n")

    # --- CANDIDATE 1: GREEDY DECODING (The Floor) ---
    print("1. Running GREEDY DECODING...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output_a = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50, 
        do_sample=False,  # Deterministic
        temperature=1.0
    )
    text_a = tokenizer.decode(output_a[0], skip_special_tokens=True)
    print("   -> Done.")

    # --- CANDIDATE 2: STANDARD SAMPLING (The Baseline) ---
    print("2. Running STANDARD SAMPLING...")
    output_b = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50, 
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.92
    )
    text_b = tokenizer.decode(output_b[0], skip_special_tokens=True)
    print("   -> Done.")

    # --- CANDIDATE 3: PHASE-SLIP (The Solution) ---
    print("3. Running PHASE SLIP SAMPLER...")
    
    sampler = PhaseSlipSampler(
        model, 
        tokenizer,
        noise_scale=0.03,           
        blend_beta=0.15,            
        logit_fusion_alpha=0.45,    
        perturbation_window=12,     
        rotation_mechanism="vector",
        dynamic_alpha=True,         
        stochastic_skip_ratio=0.0   
    )
    
    print("   [Setup] Calibrating attention heads...")
    sampler.calibrate_heads(prompt, search_layers=6)
    
    text_c = sampler.generate(prompt, max_new_tokens=50, temperature=0.65)
    print("   -> Done.")
    
    # --- THE VERDICT ---
    print("\n" + "="*80)
    print("VISUAL COMPARISON")
    print("="*80)
    
    print(f"1. GREEDY DECODING (Likely repetitive):")
    print(f"\"{text_a}\"")
    print("-" * 80)
    
    print(f"2. STANDARD SAMPLING (Wildcard):")
    print(f"\"{text_b}\"")
    print("-" * 80)
    
    print(f"3. PHASE-SLIP (Creative & Stable):")
    print(f"\"{text_c}\"")
    print("="*80)

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print("\n\n!!! CRITICAL ERROR !!!")
        print(f"The demo crashed. Error details:\n{e}")
        traceback.print_exc()
    
    input("\nDemo complete. Press Enter to close window...")