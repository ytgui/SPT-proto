# naive_gpt

## Kernel
+ using shared memory


## Features
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | decouple v1 and v2 operators |
| TODO   | finalize routed ffn forward |
| TODO   | construct MHA fine-tuning pipeline |
| TODO   | profile MHA with ablation study |


## Section 5
+ the gradient issue around PQ
+ efficient csr-based kernel / pseudo
+ improve: pre-compute indptr, multi-dim indices
