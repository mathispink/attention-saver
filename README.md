# Attention Saver

**Extract and analyze ultra-long-sequence attention matrices from HuggingFace LLMs — efficiently and at scale!**

---

## Motivation

Large language models are increasingly evaluated on long-context tasks, but extracting their full attention patterns over 100k+ tokens can quickly overwhelm GPU memory when using an eager attention implementation, and is impossible when using memory-efficient attention kernels as they never hold the full attention matrix in memory. For researchers aiming to analyze, visualize, or quantify attention in detail—across any layer or head—there's a need for a workflow that is both scalable and hackable. Attention Saver fills this gap!

**Attention Saver** provides a practical toolkit to:

- Extract raw attention matrices (for arbitrarily long sequences)
- Save directly to disk in HDF5 format, sidestepping RAM and GPU limits
- Compute and save custom per-row statistics per attention head and layer


---

## Features

- **Handles long context windows**: Works with sequences >100,000 tokens (disk space permitting)
- **Layer/head selection**: Focus on specific layers of interest
- **Custom statistics**: Plug in Python/numpy functions for per-row analysis
- **Simple interface**: Wraps around standard HuggingFace forward passes
- **No RAM bottleneck**: All results are streamed to disk row by row

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/MathisPink/attention_saver.git
```

Or clone and install locally:

```bash
git clone https://github.com/MathisPink/attention_saver.git
cd attention_saver
pip install .
```

---

## Quick Start

Here’s an example using a HuggingFace causal LLM:

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from attention_saver import AttentionSaver

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa").to("cuda")

text = "The attention matrix will be huge! " * 10000  # ~60k tokens
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with AttentionSaver(
    model=model,
    layer_ids=[4, 5, 6],
    output_file="attention_matrices.h5",
    compute_softmax=True,
    dtype="float16", # only used for saving, not for the forward pass
    save_statistic_only=False,
    row_wise_statistics=[
        lambda p: -np.sum(p * np.log2(p + 1e-9)),  # entropy
        lambda p: np.sum(np.abs(p - 1.0 / len(p))), # L1 from uniform
    ],
):
    model(**inputs)
```

---

## Accessing and Visualizing Output

Attention matrices and statistics are stored in HDF5 format for easy downstream analysis:

```python
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("attention_matrices.h5", "r") as f:
    arr = f["layer_5/batch_0/head_0"][:]
    plt.figure(figsize=(10,8))
    plt.imshow(np.log(arr + 1e-20), aspect='auto')
    plt.colorbar()
    plt.title("Attention Matrix (log scale)")
    plt.show()
```

---

## Frequently Asked Questions

**Q: Which models are supported?**  
A: Any HuggingFace causal LLM using the standard attention interfaces (eager, sdpa, flashattn, flex-attn).

**Q: Can I compute several statistics at once?**  
A: Yes—pass a list of callables to `row_wise_statistics`. Each function will populate a column in the HDF5 output.

**Q: Is there a command-line tool?**  
A: No CLI is included. All usage is via Python, making it easy to integrate into experiments and notebooks.

**Q: FlashAttention2 or custom attention implementations?**  
A: Yes, as long as the HuggingFace attention interface is respected. The method is agnostic to attention implementation.

**Q: Is sparse attention supported?**
A: No, currently, only full self-attention is supported.

---

## License

Apache 2.0

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{Pink2025attentionsaver,
  author = {Mathis Pink},
  title = {Attention Saver: Extracting ultra-long context attention matrices for HuggingFace LLMs},
  year = {2025},
  url = {https://github.com/MathisPink/attention_saver},
}
```

---

**Questions, suggestions, or issues?**  
Please open an issue or PR on GitHub.


