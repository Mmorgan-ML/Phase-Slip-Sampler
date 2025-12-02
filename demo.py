import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

def run_comparison():
    print("--- PHASE SLIP: THE CONTROL EXPERIMENT ---")
    
    # 1. Load the Brain
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 2. The Tricky Prompt
    prompt = "The scientist opened the door to the secret lab and discovered"
    print(f"\nPROMPT: '{prompt}'\n")

    # --- EXPERIMENT A: NORMAL BRAIN (Control) ---
    print("running: STANDARD GPT-2 (No Phase Slip)...")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate without any tricks
    output_a = model.generate(
        inputs.input_ids, 
        max_new_tokens=30, 
        do_sample=False # Deterministic (Greedy)
    )
    text_a = tokenizer.decode(output_a[0], skip_special_tokens=True)
    print("Done.")

    # --- EXPERIMENT B: PHASE SLIP BRAIN (Variable) ---
    print("\nrunning: PHASE SLIP SAMPLER (Your Invention)...")
    sampler = PhaseSlipSampler(model, tokenizer)
    text_b = sampler.generate(prompt, max_new_tokens=30)
    
    # --- THE VERDICT ---
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"1. NORMAL GPT-2:\n   {text_a}")
    print("-" * 40)
    print(f"2. PHASE SLIP GPT-2:\n   {text_b}")
    print("="*40)

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to close...")