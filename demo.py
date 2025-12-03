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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

def run_comparison():
    print("--- PHASE SLIP: VISUAL DEMO ---")
    
    # 1. Load the Brain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 on {device}...")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 2. The Tricky Prompt
    # This prompt often causes standard GPT-2 to loop ("The lab was a lab...")
    prompt = "The scientist opened the door to the secret lab and discovered"
    print(f"\nPROMPT: '{prompt}'\n")

    # --- EXPERIMENT A: NORMAL BRAIN (Control) ---
    print("1. Running STANDARD GPT-2 (Greedy)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output_a = model.generate(
        inputs.input_ids, 
        max_new_tokens=40, 
        do_sample=False  # Deterministic
    )
    text_a = tokenizer.decode(output_a[0], skip_special_tokens=True)
    print("   -> Done.")

    # --- EXPERIMENT B: PHASE SLIP BRAIN (Variable) ---
    print("2. Running PHASE SLIP SAMPLER...")
    # We turn on 'verbose' to see when the entropy spikes happen
    sampler = PhaseSlipSampler(model, tokenizer, confusion_threshold=3.8)
    text_b = sampler.generate(prompt, max_new_tokens=40, verbose=True)
    
    # --- THE VERDICT ---
    print("\n" + "="*40)
    print("VISUAL COMPARISON")
    print("="*40)
    print(f"STANDARD:\n{text_a}")
    print("-" * 40)
    print(f"PHASE-SLIP:\n{text_b}")
    print("="*40)

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print("\n\n!!! CRITICAL ERROR !!!")
        print(f"The demo crashed. Error details:\n{e}")
        traceback.print_exc()
    
    # This keeps the window open
    input("\nDemo complete. Press Enter to close window...")