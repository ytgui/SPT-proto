# naive_gpt


## Features
| Status | Description                          |
| ------ | ------------------------------------ |
| TODO   | construct llama fine-tuning pipeline |
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


## Scripts
+ python3 script/2-model-info.py --ckpt=.data/sheared-llama-2.7b.ckpt --tuning=lora
+ python3 script/9-profile.py --name='facebook/opt-1.3b' --tuning='lora' --module='mha' --backward
