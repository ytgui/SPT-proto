# Reproducibility Repo

## SPT: Fine-Tuning Transformer-based Language Models Efficiently with Sparsification

Transformer-based large language models (e.g., BERT and GPT) achieve great success, and fine-tuning, which tunes a pre-trained model on a task-specific dataset, is the standard practice to utilize these models for downstream tasks. However, Transformer fine-tuning has long running time and high memory consumption due to the large size of the models. We propose the SPT system to fine-tune Transformer-based models efficiently by introducing sparsity. We observe that the memory consumption of Transformer mainly comes from storing attention weights for multi-head attention (MHA), and the majority of running time is spent on feed-forward network (FFN). Thus, we design the sparse MHA module, which computes and stores only large attention weights to reduce memory consumption, and the routed FFN module, which dynamically activates a subset of model parameters for each token to reduce computation cost. We implement SPT on PyTorch and customize CUDA kernels to run sparse MHA and routed FFN efficiently. Specifically, we use product quantization to identify the large attention weights and compute attention via sparse matrix multiplication for sparse MHA. For routed FFN, we batch the tokens according to their activated model parameters for efficient computation. We conduct extensive experiments to evaluate SPT on various model configurations. The results show that SPT consistently outperforms well-optimized baselines, reducing the peak memory consumption by up to 50% and accelerating fine-tuning by up to 2.2x.


## Current Status
+ Preprint on arxiv is scheduled
