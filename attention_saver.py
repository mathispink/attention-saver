import h5py
import torch
import importlib
import logging
from typing import Callable, List, Optional, Any, Set, Dict
import numpy as np

class AttentionSaver:
    """
    Context manager for extracting attention matrices or row-wise statistics 
    from selected layers of HuggingFace LLMs, without blowing out GPU memory.

    Why?
        - Ultra-long sequence attention matrices are infeasible to keep in memory.
        - Modern efficient attention implementations that enable long-context (e.g., SDPA or flash-attn) do not support outputting attention matrices.
        - This tool saves each row directly to disk, allowing inspection of attention patterns or statistical analysis 
          on real models with >100k context.

    Usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> import torch
        >>> model_name = "meta-llama/Llama-3.2-1B"
        >>> model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> inputs = tokenizer("Long input...", return_tensors="pt").to("cuda")
        >>> with AttentionSaver(
        ...     model, layer_ids=[5], output_file="attn.h5", 
        ...     save_statistic_only=True,
        ...     row_wise_statistics=[lambda p: np.linalg.norm(p, ord=1)]  # L1 norm example (attention sparsity)
        ... ):
        ...     model(**inputs)

    Args:
        model: HuggingFace AutoModelForCausalLM or compatible.
        layer_ids: List of attention layer indices to log.
        output_file: Path to HDF5 file for output.
        dtype: Data type to save (default "float16").
        compute_softmax: If True, apply softmax to logits before saving.
        save_statistic_only: If True, save only user-specified row-wise statistics, not full matrices.
        row_wise_statistics: List of callables, each mapping a 1D np.ndarray (probabilities) to float. 
            If None, statistics are not computed.
        causal_LLM: If True, apply causal masking to logits (default True).
        verbose: If True, print/save progress information.

    Raises:
        ValueError: If `row_wise_statistics` contains non-callables.
    """
    def __init__(
        self,
        model: Any,
        layer_ids: List[int],
        output_file: str,
        dtype: str = "float16",
        compute_softmax: bool = True,
        save_statistic_only: bool = False,
        row_wise_statistics: Optional[List[Callable[[np.ndarray], float]]] = None,
        causal_LLM: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.modeling = importlib.import_module(model.__module__)
        self.layer_ids: Set[int] = set(layer_ids)
        self.output_file = output_file
        self.dtype = dtype
        self.compute_softmax = compute_softmax if not save_statistic_only else False
        self.save_statistic_only = save_statistic_only
        self.causal_LLM = causal_LLM
        self.verbose = verbose


        # Handle statistics argument and check validity
        if row_wise_statistics is None:
            self.row_wise_statistics: List[Callable[[np.ndarray], float]] = []
        elif isinstance(row_wise_statistics, (list, tuple)):
            if not all(callable(f) for f in row_wise_statistics):
                raise ValueError("All elements of `row_wise_statistics` must be callable functions.")
            self.row_wise_statistics = list(row_wise_statistics)
        else:
            raise ValueError("row_wise_statistics must be a list/tuple of callables or None.")

        self.file: Optional[h5py.File] = None
        self._original_functions: Dict[str, Callable] = {}

    def __enter__(self) -> "AttentionSaver":
        self.file = h5py.File(self.output_file, "w")
        self._original_functions = {}

        for name, orig_func in self.modeling.ALL_ATTENTION_FUNCTIONS.items():
            def make_wrapper(f=orig_func, name=name):
                def wrapped(
                    attn_self,
                    query_states: torch.Tensor,
                    key_states: torch.Tensor,
                    value_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    **kwargs
                ):
                    layer_idx = getattr(attn_self, "layer_idx", None)
                    if (layer_idx is not None) and (layer_idx in self.layer_ids):
                        with torch.no_grad():
                            qs = query_states.clone()
                            ks = key_states.clone()
                            bsz, n_heads, seq_len, head_dim = qs.shape
                            if ks.shape[1] != n_heads:
                                repeat_factor = n_heads // ks.shape[1]
                                ks = ks.repeat_interleave(repeat_factor, dim=1)

                            sqrt_hd = head_dim ** 0.5

                            for batch_idx in range(bsz):
                                for head_idx in range(n_heads):
                                    group_name = f"layer_{layer_idx}/batch_{batch_idx}/head_{head_idx}"
                                    if self.verbose:
                                        logging.info(
                                            f"Saving {'attention matrix' if not self.save_statistic_only else 'row-wise stats'} for {group_name}"
                                        )
                                    data_shape = (seq_len, seq_len) if not self.save_statistic_only else (seq_len, len(self.row_wise_statistics))
                                    dset = self.file.create_dataset(
                                        group_name,
                                        data_shape,
                                        dtype=self.dtype
                                    )
                                    q = qs[batch_idx, head_idx]    # (seq_len, head_dim)
                                    k = ks[batch_idx, head_idx]    # (seq_len, head_dim)
                                    for row_idx in range(seq_len):
                                        q_row = q[row_idx].unsqueeze(0)     # (1, head_dim)
                                        logits_row = (q_row @ k.T) / sqrt_hd
                                        logits_row = logits_row.squeeze(0)     # (seq_len,)
                                        if self.causal_LLM:
                                            logits_row[row_idx+1:] = -torch.inf
                                        if self.compute_softmax:
                                            probs_row = torch.softmax(logits_row, dim=0)
                                        else:
                                            probs_row = logits_row
                                        if not self.save_statistic_only:
                                            dset[row_idx, :] = probs_row.cpu().float().numpy()
                                        else:
                                            for i, row_wise_statistic in enumerate(self.row_wise_statistics):
                                                dset[row_idx, i] = row_wise_statistic(
                                                    probs_row.cpu().float().numpy()
                                                )
                                    self.file.flush()
                    return f(attn_self, query_states, key_states, value_states, attention_mask, **kwargs)
                return wrapped
            self._original_functions[name] = orig_func
            self.modeling.ALL_ATTENTION_FUNCTIONS[name] = make_wrapper()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore default attention functions
        for name, orig_func in self._original_functions.items():
            self.modeling.ALL_ATTENTION_FUNCTIONS[name] = orig_func
        self._original_functions = {}
        if self.file is not None:
            self.file.close()