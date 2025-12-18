# Phase-Slip: Latent Perturbation for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Concept
Standard sampling methods (Temperature, Top-K) introduce randomness at the very last step of generation: the output logits. While effective, this "surface-level" noise often leads to perplexity spikes- moments where the model chooses a creative word that breaks the logical flow of the sentence, leading to hallucinations or grammar failures.

Phase-Slip Sampling is a stochastic intervention architecture that operates on the KV cache of the model. Instead of forcing the model to pick a random word, Phase-Slip gently rotates the semantic vectors of the context window, effectively asking the model: *"How would you finish this sentence if you looked at it from a slightly different perspective?"*

The result is a sampler that achieves the creativity of high temperatures with significantly lower perplexity.

## Installation

### For Users
You can install the package directly from PyPI:

```bash
pip install phase-slip-sampler
```

### For Developers
If you are cloning this repository for local development or research:

```bash
git clone https://github.com/Mmorgan-ML/phase-slip-sampler.git
cd phase-slip-sampler
pip install -r requirements.txt
```

> **Note:** While the package name is `phase-slip-sampler`, the Python module is named `phase_slip`.

## Usage

### Python Import
Phase-Slip works as a wrapper around Hugging Face Transformers models.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

# 1. Load your model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 2. Initialize the Sampler
# The default settings are tuned for "Coherent Creativity"
sampler = PhaseSlipSampler(
    model, 
    tokenizer,
    noise_scale=0.03,           # Magnitude of vector rotation
    logit_fusion_alpha=0.45,    # Strength of the "Clean" anchor
    perturbation_window=12      # Effective semantic horizon
)

# 3. Calibrate (Optional but Recommended)
# Scans attention heads to find the "Creative" heads vs "Grammar" heads
sampler.calibrate_heads("Once upon a time", search_layers=6)

# 4. Generate
text = sampler.generate("The ancient door creaked open", max_new_tokens=50, temperature=0.65)
print(text)
```

### Running the Demo
To see the sampler in action:
```bash
python demo.py
```

### Benchmarking
To replicate the research findings found in this README:
```bash
python benchmark.py
```

## Mechanism of Action

Phase-Slip is significantly more complex than standard sampling. For every token generated, the architecture performs a dual-path forward pass:

1. Automatic Head Calibration: Before sampling begins, a scanning utility profiles attention heads to identify those correlated with semantic exploration (“creative” heads) versus those responsible for syntax, logic, and factual integrity (“structural” heads). Only the creative heads are marked as eligible for perturbation; structural heads are explicitly excluded.
2.  Copy the KV cache: The sampler creates a copy of the Key-Value Cache.
3.  Orthonormal Rotation: Instead of adding destructive Gaussian noise (which breaks the manifold), the sampler applies a geometric rotation to the Value vectors in specific attention heads. This preserves the magnitude of the signal while shifting the semantic nuance.
4.  The Pertubed Pass: The model performs a forward pass using this perturbed memory to generate a set of "Creative Logits."
5.  Logit Fusion: These creative logits are mathematically fused with the logits from the unperturbed memory using a dynamic alpha gate.
    *   *If the model is confident (Low Entropy),* the unperturbed pass dominates.
    *   *If the model is uncertain (High Entropy),* the perturbed path is taken.
6.  Discarding the perturbed tokens: Once the token is chosen, the perturbed token is discarded. The model "remembers" saying the creative word, but "forgets" the neurological state that caused it. This prevents errors from cascading.

## Empirical Evidence

Benchmarks performed on `gpt2` (Small) over 5 diverse prompts (40 rounds each, N=200) demonstrate that Phase-Slip occupies a unique niche: High Stability Creativity.

### 1. The "Coherence Gap" (Quantitative Data)

| Method | Diversity (Higher is Better) | Perplexity (Lower is Better) | Speed (Tok/s) |
| :--- | :--- | :--- | :--- |
| **Greedy Decoding** (Control) | 0.09 ± 0.01 | 1.29 ± 0.02 | 20.4 |
| **Standard Sampling** (Baseline) | **0.37** ± 0.14 | 4.49 ± 1.83 | **18.6** |
| **Phase-Slip** (Strong Anchor) | 0.32 ± 0.15 | **3.66** ± **1.65** | 6.8 |

*Data collected via `benchmark.py` (v1.0.1) on 2025.12.13.*

**Analysis:**
*   **Perplexity Win:** Phase-Slip achieves a Perplexity of 3.66 compared to Standard Sampling's 4.49. This represents an ~18.5% improvement, with a more narrow standard deviation (1.65) vs Standard Sampling (1.83).
*   **Diversity Trade-off:** We sacrifice a small amount of diversity (0.32 vs 0.37) to achieve this stability. The model is less likely to produce "wild" hallucinations.

### 2. The "Loop Breaker" Effect (Qualitative)
*Legacy test demonstrating the core de-stagnation capability.*

**Prompt:** *"The research paper described the finding that the"*

| Method | Output Snippet | Behavior |
|--------|----------------|----------|
| **Greedy Decoding** | "...brain's ability to process information... brain... brain is able to process information..." | **FAILURE:** Classic logic loop. The model repeats phrases endlessly. |
| **Phase-Slip** | "...children with ADHD make less convulsions... 'implicated disorder' of high-level students..." | **SUCCESS:** The vector rotation forces the model out of the local probability minimum, generating new concepts. |

## Limitations & Trade-Offs

Phase-Slip is a research architecture. It is not a drop-in replacement for every use case.

1.  The Speed Penalty: Because Phase-Slip requires two forward passes (one Clean, one Perturbed) plus Python-side vector math, it runs at approximately 35-40% the speed of Standard Sampling. It is not recommended for high-throughput production environments.
2.  Awkward phrasing: On very small models (like GPT-2), the perturbations can sometimes lead to collocation errors (e.g., "A room filled with a man" instead of "containing a man"). This effect may diminish with larger model sizes (Llama-3, Mistral).

## License

MIT


