# naive_gpt


## Features
| Status | Description                          |
| ------ | ------------------------------------ |
| TODO   | construct LoRA + RoutedFFN           |
| TODO   | profile RoutedFFN for ablation study |
| TODO   | keep v1 and v2 attention block       |
| TODO   | construct LoRA + SparseMHA           |
| TODO   | construct MHA fine-tuning pipeline   |
| TODO   | profile MHA with ablation study      |
| TODO   | prepare MMLU evaluation pipeline     |


## Known Issues
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | cuda 11.x has no half precision SDDMM |
| TODO   | Llama 2 is not available in CN or HK  |
| TODO   | Llama-3b has an unaligned d_head=100  |


## Section 5
+ the gradient issue around PQ
+ efficient csr-based kernel / pseudo
+ improve: pre-compute indptr, multi-dim indices
