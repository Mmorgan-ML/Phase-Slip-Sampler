# Phase-Slip: Latent Perturbation for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##Warning
Current implementation causes massive hallucinations and high PPL after escaping the loop. This project is a work in progress.

## The Concept
Neural text generation often collapses into repetitive loops ("degenerate repetition") because the model falls into a deep local minimum of probability—it becomes too confident in its own redundant output. Standard penalties operate on the *output* (logits), often leading to grammatical fracturing.

**Phase-Slip** is a stochastic intervention sampler that operates on the **internal memory** of the model.
1.  **Monitor:** It tracks the **Shannon Entropy** of the model in real-time.
2.  **Detect:** It identifies **"Stagnation States"**—periods where entropy drops below a threshold for too long, indicating a loop.
3.  **Perturb:** It injects non-destructive Gaussian noise directly into the **Key-Value (KV) Cache**.

This "Latent Shock" effectively shakes the model's memory, forcing it to hallucinate a new context and break out of the local minimum.

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
To use the sampler in your own code:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from phase_slip.sampler import PhaseSlipSampler

model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize the Sampler
# stagnation_threshold: How confident is "too confident"? (Lower = stricter)
# noise_scale: Magnitude of the memory shock (Higher = more chaotic)
sampler = PhaseSlipSampler(
    model, 
    tokenizer, 
    stagnation_threshold=0.6, 
    patience=3,
    noise_scale=0.1
)

text = sampler.generate("The scientific method is", max_new_tokens=50)
print(text)
```

### Running the Demo
To see the "Thermal Shock" in action and visualize the divergence from greedy decoding:
```bash
python demo.py
```

### Benchmarking
To statistically compare Phase-Slip against Greedy Decoding and Standard Sampling:
```bash
python benchmark.py
```

## Empirical Evidence

Benchmarks performed on `gpt2` (Small) demonstrate that Phase-Slip effectively shatters repetition loops, achieving higher vocabulary diversity than even standard temperature sampling.

### 1. The "Loop Breaker" Test
**Prompt:** *"The research paper described the finding that the"*

| Method | Output Snippet | Behavior |
|--------|----------------|----------|
| **Greedy Decoding** | "...brain's ability to process information... brain... brain is able to process information..." | **FAILURE:** Classic logic loop. The model repeats "brain" and "process information" endlessly. |
| **Phase-Slip** | "...children with ADHD make less convulsions... 'implicated disorder' of high-level students..." | **SUCCESS:** Detected low entropy (stagnation), injected KV noise, and forced a complete semantic divergence. |

### 2. Vocabulary Diversity Score (n=5 rounds)
*Score based on unique word count ratio. Higher is better.*

| Method | Avg Score | Consistency |
|--------|-----------|-------------|
| **Greedy Decoding** | `0.26` | Locked in loops. Zero creativity. |
| **Standard Sampling** | `0.65` | High variance (ranged from 0.25 to 0.81). |
| **Phase-Slip** | **`0.81`** | **Consistently high diversity (>0.75).** |

*Data collected via `benchmark.py` on 2025.12.03.*

## Calibration & Limitations

This method balances **Repetition** against **Coherence**. 

*   **Noise Scale:** Controls the magnitude of the "Shock." 
    *   Low (`0.05`): Subtle nudges. Keeps grammar intact but might not break strong loops.
    *   High (`0.15+`): Strong divergences. Can lead to "dream-like" or nonsensical transitions (e.g., breaking syntax).
*   **Stagnation Threshold:** Controls the trigger sensitivity.
    *   `0.6` is a recommended starting point for GPT-2/Llama.
    *   Setting this too high (e.g., `0.8`) will trigger shocks constantly, leading to incoherence.

## Project Structure
*   `phase_slip/`: The source code for the sampler.
    *   `sampler.py`: Contains the `latent_perturbation` logic.
*   `demo.py`: A visual comparison script.
*   `benchmark.py`: A statistical tool to measure vocabulary diversity.

## License

MIT
