# naive_gpt

## Kernel
+ using shared memory


## Features
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | bump models.PQ into here, refine it   |
| TODO   | optimize N-dimentional PQ estimator   |
| TODO   | optimize and prune FFN by blkmm       |
| TODO   | finish phase-2 tuning by using sddmm  |
| TODO   | build complete fine-tuning pipeline   |
| TODO   | evaluate MMLU score for OPT and LLAMA |


## Timing (seq=512, bsz=16)
| Stage             | Time |
| ----------------- | ---- |
| quantizer.train   | 0.05 |
| quantizer.encode  | 0.20 |
| torch.topk        | 0.01 |
| kernel.sparse_mha | 0.03 |


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
  + fused sparse mha
  + top-k selection
  + sparse ffn


## Evaluation
+ GPT
  + MMLU: knowledge intensive multiple choice
+ BERT
  + GLUE: pip install evaluate (huggingface)
+ Dataset
  + (PPL) Alpaca-52k: 42 MB
  + (Accuracy) MMLU: 166 MB
  + (PPL) LaMini-2.6M: 700 MB
