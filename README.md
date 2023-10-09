# naive_gpt


## Features
| Status | Description                        |
| ------ | ---------------------------------- |
| TODO   | keep v1 and v2 attention block     |
| TODO   | construct MHA fine-tuning pipeline |
| TODO   | profile MHA with ablation study    |
| TODO   | load and convert LLaMa2 layers     |
| TODO   | build and test LLaMa2 model (NT)   |


## Known Issues
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | cuda 11.x has no half precision SDDMM |


## Section 5
+ the gradient issue around PQ
+ efficient csr-based kernel / pseudo
+ improve: pre-compute indptr, multi-dim indices
