# Phase-Slip-Sampler: Entropy-Guided Thermal Shock for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Concept
Standard LLM generation forces a choice between rigidity (Greedy Decoding) and randomness (Standard Sampling). 

**Phase-Slip** is a dynamic inference sampler that utilizes the **Thermodynamics of Attention**.
1.  **Monitor:** It calculates the Shannon Entropy (uncertainty) of the model at every step.
2.  **Trigger:** When confusion spikes above a threshold, it triggers a "Phase Slip."
3.  **Shock:** It injects Gaussian noise ("Heat") directly into the Key-Value memory tensors and temporarily spikes the sampling temperature.

This simulated annealing forces the model to "shake loose" from local minima (repetitive loops) only when necessary, preserving structure when the model is confident.

## Installation

You can install the package directly from PyPI:

```bash
pip install phase-slip-sampler
```

Alternatively, for local development from the source repository:
```bash
pip install -r requirements.txt
```

> **Important:** While the package name is `phase-slip-sampler`, the Python module is named `phase_slip`.

## Usage

### Python Import
To use the sampler in your own code, remember to use the underscore naming convention:

```python
import phase_slip

sampler = phase_slip.Sampler(...)
```

### Quick Demo
To see the sampler in action and watch it trigger "Thermal Shocks" in real-time:
```bash
python demo.py
```

### Benchmarking
To statistically compare Phase-Slip against Greedy Decoding and Standard Sampling:
```bash
python benchmark.py
```

## Evidence
Benchmarks performed on `gpt2` (Small) demonstrate that Phase-Slip effectively breaks repetition loops without requiring constant high-temperature sampling.

### 1. Vocabulary Diversity (n=5 rounds)
*Score based on unique word count ratio. Higher is better.*

| Method | Score | Behavior |
|--------|-------|----------|
| **Greedy Decoding** | `0.26` | Highly repetitive. Stuck in loops. |
| **Standard Sampling** | `0.59` | High variance. Good for creative writing. |
| **Phase-Slip** | **`0.60`** | **Matches standard sampling diversity but uses an adaptive trigger.** |

### 2. The "Loop Breaker" Test
**Prompt:** *"The scientist opened the door to the secret lab and discovered"*

**Standard GPT-2 (Greedy):**
> "...that the lab was a laboratory for the research of the human brain. 'I was very surprised to find out that the lab was a laboratory..."
> *(FAILURE: Stuck in a logic loop)*

**Phase-Slip GPT-2:**
> "...that how the human body works was influenced by the environment. West's research was published in the journal Heart. 'The microwave is a very important part..."
> *(SUCCESS: Detected entropy spike, injected heat, and forced a semantic divergence)*

## Project Structure
*   `phase_slip/`: The source code for the sampler.
*   `demo.py`: A visual comparison script to see the thermal shocks.
*   `benchmark.py`: A statistical tool to measure vocabulary diversity.

## License

MIT
