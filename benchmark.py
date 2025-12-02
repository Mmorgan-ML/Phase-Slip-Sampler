# benchmark.py

"""
 Original Author: Michael Christian Morgan
 2025.12.03
 Github: https://github.com/Mmorgan-ML
 Twitter: @Mmorgan_ML
 Email: mmorgankorea@gmail.com
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler  # Importing from your existing file

def count_unique_words(text):
    # This counts how many different words are used.
    # More unique words = More creativity.
    words = text.lower().replace(".", "").replace(",", "").split()
    return len(set(words))

def run_benchmark():
    print("--- STARTING CREATIVITY BENCHMARK ---")
    
    # 1. Load the Brain
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # A boring prompt that usually makes AI repeat itself
    prompt = "The scientific method is a process that"
    
    print(f"Prompt: '{prompt}'")
    print("Running 5 rounds to compare...\n")

    # --- CONTROL GROUP (Standard) ---
    control_scores = []
    print("1. Testing Standard AI (Control)...")
    for i in range(5):
        inputs = tokenizer(prompt, return_tensors="pt")
        # Standard generation
        output = model.generate(inputs.input_ids, max_new_tokens=40, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        score = count_unique_words(text)
        control_scores.append(score)
        print(f"   Round {i+1} Unique Words: {score}")

    avg_control = sum(control_scores) / len(control_scores)

    # --- EXPERIMENTAL GROUP (Phase Slip) ---
    phase_scores = []
    print("\n2. Testing Phase-Slip AI (Your Invention)...")
    
    for i in range(5):
        # We start a fresh sampler every time
        sampler = PhaseSlipSampler(model, tokenizer) 
        text = sampler.generate(prompt, max_new_tokens=40)
        
        score = count_unique_words(text)
        phase_scores.append(score)
        print(f"   Round {i+1} Unique Words: {score}")

    avg_phase = sum(phase_scores) / len(phase_scores)

    # --- THE RESULTS ---
    print("\n" + "="*40)
    print("FINAL SCORECARD")
    print("="*40)
    print(f"Standard AI Score:   {avg_control:.1f}")
    print(f"Phase-Slip AI Score: {avg_phase:.1f}")
    print("-" * 40)
    
    if avg_phase > avg_control:
        print("CONCLUSION: SUCCESS! Phase-Slip is more creative.")
    else:
        print("CONCLUSION: No significant change.")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
    input("Press Enter to close...")