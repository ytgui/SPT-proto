# naive_gpt

## Features
| Status | Description                             |
| ------ | --------------------------------------- |
| TODO   | refine tuning stages                    |
| TODO   | decouple FFN tuning script with lr 1e-6 |
| TODO   | fix the training interval issues on MHA |
| TODO   | validate the efficiency again           |
| TODO   | decouple 2 stages of MHA tuning script  |


## Known Issues
| Status | Description                                |
| ------ | ------------------------------------------ |
| TODO   | cuda 11.x has no half precision SDDMM      |
| TODO   | Llama 2 is not available in CN or HK       |
| TODO   | Llama-3b has an unaligned d_head=100       |
| TODO   | PyTorch didn't implement cdist for float16 |


## Scripts
+ python3 script/2-model-info.py --ckpt=.data/sheared-llama-2.7b.ckpt --tuning=lora
+ python3 script/9-profile.py --name='opt-2048' --tuning='lora' --module='mha' --backward
