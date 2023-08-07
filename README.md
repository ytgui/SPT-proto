# naive_gpt

## Kernel
+ using shared memory
+ support batch_size and n_heads


## Features
| Status | Description                                      |
| ------ | ------------------------------------------------ |
| DONE   | load_1: dump transformers into checkpoint        |
| DONE   | test_1: load checkpoint and compare results      |
| DONE   | test_2: upgrade pre-trained model with LoRA      |
| DONE   | model_1: dense DefaultOPTModel implementation    |
| DONE   | model_2: LoRA ModuleHandler for model_1          |
| DONE   | loader_1: wikitext-103 dataset loader module     |
| DONE   | loader_2: alpaca-cleaned dataset loader module   |
| DONE   | train_1: insert k-means pq layers into GPT model |
| DONE   | train_2: make opt ready for dense fine-tuning    |
| TODO   | train_3: build complete fine-tuning pipeline     |
| TODO   | evaluate MMLU score, [llama, opt] * [ppl, mmlu]  |


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
