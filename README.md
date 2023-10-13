# naive_gpt


## Features
| Status | Description                           |
| ------ | ------------------------------------- |
| DONE   | construct module upgrader for LoRA    |
| DONE   | construct LoRA + VanillaBlocks        |
| DONE   | construct LoRA + RoutedFFN            |
| TODO   | profile LoRA + FFN for ablation       |
| TODO   | profile LoRA + RoutedFFN for ablation |
| TODO   | keep v1 and v2 attention block        |
| TODO   | construct LoRA + SparseMHA            |
| TODO   | construct MHA fine-tuning pipeline    |
| TODO   | profile MHA with ablation study       |
| TODO   | prepare MMLU evaluation pipeline      |


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
