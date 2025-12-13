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
import time
import random
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

# Fix: Silence huggingface warnings
transformers.logging.set_verbosity_error()

def calculate_diversity_score(text):
    words = text.lower().replace(".", "").replace(",", "").split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)

def calculate_perplexity(model, tokenizer, full_text, prompt_text):
    device = model.device
    enc_prompt = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    enc_full = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    if enc_full.shape[1] <= enc_prompt.shape[1]: return 1000.0
    labels = enc_full.clone()
    prompt_len = enc_prompt.shape[1]
    labels[:, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(enc_full, labels=labels)
        loss = outputs.loss
    return torch.exp(loss).item()

def run_benchmark():
    print("--- AUTOMATED PHASE-SLIP: PRODUCTION BENCHMARK (v1.0.1) ---")
    print("Protocol: Fixed-Seed Paired Testing | Mean +/- Std Dev | Precise Timing")
    print("Configuration: 5 Prompts | 40 Rounds | 200 Tokens (Full Rigor)")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   > Device: {device}")
        
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Fix for GPT-2 Padding Issue
        model.config.pad_token_id = model.config.eos_token_id
        
        # --- 1. CALIBRATION STEP ---
        print("\n>> Running Head Calibration (Scanning last 6 layers)...")
        calib_sampler = PhaseSlipSampler(model, tokenizer)
        calib_prompt = "Once upon a time in a land where magic and science merged,"
        target_heads = calib_sampler.calibrate_heads(calib_prompt, search_layers=6)
        print(f"   > Calibration Complete. Target Heads: {target_heads}")
        
        # FULL RESEARCH SUITE (5 Prompts)
        prompts = [
            "The scientific method is a process that",              # Factual / Logical
            "Once upon a time in a land where magic and science",   # Narrative / Creative
            "The meaning of the void is",                           # Abstract / High Entropy
            "To fix the engine, you must first",                    # Instructional / Procedural
            "The system detected a failure in the"                  # Technical / Systemic
        ]
        
        configs = []
        
        # --- THE BIG THREE CANDIDATES ---
        
        # 1. Greedy Decoding (The Floor)
        configs.append({
            "name": "Greedy Decoding (Control)",
            "type": "native",
            "kwargs": {"do_sample": False, "temperature": 1.0}
        })

        # 2. Standard Sampling (The Baseline)
        configs.append({
            "name": "Standard Sampling (Baseline)", 
            "type": "native", 
            "kwargs": {"do_sample": True, "temperature": 0.8, "top_k": 40, "top_p": 0.92}
        })
        
        # 3. Phase-Slip (The Solution)
        configs.append({
            "name": "Phase-Slip (Strong Anchor, T=0.65)",
            "type": "phase_slip",
            "kwargs": {
                "noise_scale": 0.03, "blend_beta": 0.15, "logit_fusion_alpha": 0.45,
                "target_heads": target_heads, "rotation_mechanism": "vector", "dynamic_alpha": True,
                "perturbation_window": 12, "stochastic_skip_ratio": 0.0, "temperature": 0.65
            }
        })

        results_table = []
        
        # FULL RIGOR SETTINGS
        rounds_per_prompt = 40 
        max_new_tokens = 200   

        # --- EXECUTION LOOP ---
        for config in configs:
            print(f"\n>> Testing Configuration: {config['name']}")
            div_list = []
            ppl_list = []
            score_list = []
            speed_list = []
            
            for p_idx, prompt in enumerate(prompts):
                print(f"   Processing Prompt {p_idx+1}/5 ", end="", flush=True)
                for r in range(rounds_per_prompt):
                    if r % 5 == 0: print(".", end="", flush=True)
                    
                    # --- FIXED SEED PROTOCOL ---
                    seed = p_idx * 1000 + r
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    t0 = time.time()
                    
                    if config["type"] == "native":
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        output_ids = model.generate(
                            inputs.input_ids, 
                            attention_mask=inputs.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=max_new_tokens, 
                            **config["kwargs"]
                        )
                        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        n_input = inputs.input_ids.shape[1]
                        generated_count = output_ids.shape[1] - n_input
                    
                    elif config["type"] == "phase_slip":
                        gen_temp = config["kwargs"].get("temperature", 0.8)
                        init_params = {k:v for k,v in config["kwargs"].items() if k != "temperature"}
                        
                        params = {
                            "mask_prompt": True,
                            "shock_temperature_factor": 1.0,
                            "speculative_candidates": 0,
                            "noise_type": "ortho_rotation",
                            "perturbation_window": 12
                        }
                        params.update(init_params)
                        
                        sampler = PhaseSlipSampler(model, tokenizer, **params)
                        text = sampler.generate(prompt, max_new_tokens=max_new_tokens, temperature=gen_temp)
                        
                        full_ids = tokenizer.encode(text)
                        prompt_ids = tokenizer.encode(prompt)
                        generated_count = max(0, len(full_ids) - len(prompt_ids))
                    
                    t1 = time.time()
                    duration = t1 - t0
                    speed = generated_count / (duration + 1e-9)
                    
                    div = calculate_diversity_score(text)
                    ppl = calculate_perplexity(model, tokenizer, text, prompt)
                    
                    safe_ppl = max(ppl, 1.0001)
                    score = div / np.log(safe_ppl)
                    
                    div_list.append(div)
                    ppl_list.append(ppl)
                    score_list.append(score)
                    speed_list.append(speed)
                print(" Done.")

            div_mean = np.mean(div_list)
            div_std = np.std(div_list)
            ppl_mean = np.mean(ppl_list)
            ppl_std = np.std(ppl_list)
            score_mean = np.mean(score_list)
            score_std = np.std(score_list)
            speed_mean = np.mean(speed_list)
            
            results_table.append({
                "Method": config["name"],
                "Div_Mean": div_mean, "Div_Std": div_std,
                "PPL_Mean": ppl_mean, "PPL_Std": ppl_std,
                "Score_Mean": score_mean, "Score_Std": score_std,
                "Speed": speed_mean
            })
            print(f"   -> Score: {score_mean:.3f} +/- {score_std:.3f}")

        # --- REPORT ---
        print("\n" + "="*120)
        print(f"{'METHOD':<45} | {'DIV (Avg ± Std)':<18} | {'PPL (Avg ± Std)':<18} | {'SCORE (Avg ± Std)':<18} | {'SPEED'}")
        print("-" * 120)
        md_table = "| Method | DIV (Avg ± Std) | PPL (Avg ± Std) | SCORE (Avg ± Std) | Speed (T/s) |\n|---|---|---|---|---|\n"
        for res in results_table:
            div_str = f"{res['Div_Mean']:.2f} ± {res['Div_Std']:.2f}"
            ppl_str = f"{res['PPL_Mean']:.2f} ± {res['PPL_Std']:.2f}"
            score_str = f"{res['Score_Mean']:.3f} ± {res['Score_Std']:.3f}"
            print(f"{res['Method']:<45} | {div_str:<18} | {ppl_str:<18} | {score_str:<18} | {res['Speed']:.1f}")
            md_row = f"| **{res['Method']}** | `{div_str}` | `{ppl_str}` | **`{score_str}`** | `{res['Speed']:.1f}` |\n"
            md_table += md_row
        print("="*120)
        print("\n--- COPY FOR README.MD ---")
        print(md_table)
        print("--------------------------")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
    input("\nBenchmark complete...")