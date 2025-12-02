# Phase-Slip: Entropy-Guided Thermal Shock for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Concept
Standard LLM generation (Greedy Decoding) is deterministic. When a model falls into a repetitive loop or a "hallucination rut," it often stays there because the probability of the next token in the loop is locally high.

**Phase-Slip** is a dynamic inference sampler that treats the LLM's memory (KV Cache) as a thermodynamic system. 
1.  **Monitor:** It calculates the entropy (confusion) of the model at every step.
2.  **Trigger:** When confusion spikes above a threshold, it triggers a "Phase Slip."
3.  **Shock:** It injects Gaussian noise ("Heat") directly into the Key-Value memory tensors. 

This simulated annealing forces the model to "shake loose" from its current trajectory and explore new semantic paths.

## The Evidence
Phase-Slip prevents LLMs from getting stuck in "boring" or repetitive loops by forcing them to diverge when entropy spikes.

**Prompt:** *"The scientist opened the door to the secret lab and discovered"*

| Method | Output | Analysis |
|--------|--------|----------|
| **Standard GPT-2** | *"...that the lab was a laboratory for the research of the human brain. 'I was very surprised to find out that the lab was a laboratory..."* | **Failure.** Stuck in a repetitive logic loop. |
| **Phase-Slip** | *"...that tens of thousands of bells were ringing. Silent, he turned to Mithra and said, 'Aye, I'm a man of Nep..."* | **Success.** completely broke the loop and hallucinated a novel, creative narrative path. |

*By injecting noise and spiking temperature only during high-confusion states, Phase-Slip acts as an automated "Writer's Block" cure.*

## Project Structure
*   `phase_slip/`: The source code for the sampler.
*   `demo.py`: A script to see the Phase Slip in action on specific prompts.
*   `benchmark.py`: A tool to measure vocabulary diversity vs standard generation.

## Usage
1. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo: 
   ```bash
   python demo.py
   ```

## License
MIT