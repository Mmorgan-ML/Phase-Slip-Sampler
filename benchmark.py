# benchmark.py

"""
 Original Author: Michael Christian Morgan
 2025.12.03
 Github: https://github.com/Mmorgan-ML
 Twitter: @Mmorgan_ML
 Email: mmorgankorea@gmail.com
"""

import torch
import numpy as np
import sys
import traceback
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

def calculate_diversity_score(text):
    words = text.lower().replace(".", "").replace(",", "").split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)

def run_benchmark():
    print("--- STARTING CREATIVITY BENCHMARK ---")
    
    try:
        # Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        prompt = "The scientific method is a process that"
        print(f"Prompt: '{prompt}'")
        
        rounds = 5
        max_tokens = 50

        # --- 1. GREEDY ---
        print("\n1. Control Group A: Greedy Decoding...")
        greedy_scores = []
        for i in range(rounds):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(inputs.input_ids, max_new_tokens=max_tokens, do_sample=False)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            score = calculate_diversity_score(text)
            greedy_scores.append(score)
            print(f"   Round {i+1}: {score:.2f}")

        # --- 2. STANDARD SAMPLING ---
        print("\n2. Control Group B: Standard Sampling...")
        sample_scores = []
        for i in range(rounds):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                inputs.input_ids, max_new_tokens=max_tokens, 
                do_sample=True, temperature=0.7, top_k=50
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            score = calculate_diversity_score(text)
            sample_scores.append(score)
            print(f"   Round {i+1}: {score:.2f}")

        # --- 3. PHASE SLIP ---
        print("\n3. Experimental Group: Phase-Slip...")
        phase_scores = []
        for i in range(rounds):
            sampler = PhaseSlipSampler(model, tokenizer, confusion_threshold=3.5) 
            text = sampler.generate(prompt, max_new_tokens=max_tokens)
            score = calculate_diversity_score(text)
            phase_scores.append(score)
            print(f"   Round {i+1}: {score:.2f}")

        # --- RESULTS ---
        print("\n" + "="*40)
        print("FINAL SCORECARD")
        print("="*40)
        print(f"Greedy:      {np.mean(greedy_scores):.2f}")
        print(f"Standard:    {np.mean(sample_scores):.2f}")
        print(f"Phase-Slip:  {np.mean(phase_scores):.2f}")
        print("-" * 40)

    except Exception as e:
        print("\n\n!!! CRITICAL ERROR !!!")
        print(f"The benchmark crashed. Error details:\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
    # This line forces the window to stay open
    input("\nBenchmark complete. Press Enter to close window...")