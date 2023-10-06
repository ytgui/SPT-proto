# naive_gpt


## Features
| Status | Description                        |
| ------ | ---------------------------------- |
| TODO   | decouple v1 and v2 operators       |
| TODO   | finalize routed ffn forward        |
| TODO   | profile MHA v.s. FFN               |
| TODO   | construct MHA fine-tuning pipeline |
| TODO   | profile MHA with ablation study    |
| TODO   | half precision for sddmm, spmm     |
| TODO   | half precision for custom kernels  |



## Section 5
+ the gradient issue around PQ
+ efficient csr-based kernel / pseudo
+ improve: pre-compute indptr, multi-dim indices
