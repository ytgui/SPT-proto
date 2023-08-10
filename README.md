# naive_gpt

## Kernel
+ using shared memory
+ support batch_size and n_heads


## Features
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | build complete fine-tuning pipeline   |
| TODO   | evaluate MMLU score for OPT and LLAMA |


## Section 5
+ the gradient issue around PQ
+ efficient csr-based kernel / pseudo
+ improve: pre-compute indptr, multi-dim indices


## Pipeline
+ Module Upgrader
  + insert LoRA layers
  + perform basic fusion
  + insert l2 regulation
+ Sparse Analyser
  + gradually replace by sparse op
+ Sparse Runtime
  + highly optimized kernels


## Evaluation
+ GPT
  + MMLU: knowledge intensive multiple choice
+ BERT
  + GLUE: pip install evaluate (huggingface)
+ Dataset
  + (PPL) Alpaca-52k: 42 MB
  + (Accuracy) MMLU: 166 MB
  + (PPL) LaMini-2.6M: 700 MB
