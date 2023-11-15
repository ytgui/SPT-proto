# naive_gpt

## Features
| Status | Description                             |
| ------ | --------------------------------------- |
| TODO   | refine tuning stages                    |
| TODO   | decouple MHA and FFN tuning script      |
| TODO   | fix the training interval issues on MHA |
| TODO   | validate the efficiency again           |


## Known Issues
| Status | Description                           |
| ------ | ------------------------------------- |
| TODO   | cuda 11.x has no half precision SDDMM |
| TODO   | Llama 2 is not available in CN or HK  |
| TODO   | Llama-3b has an unaligned d_head=100  |


## Scripts
+ python3 script/2-model-info.py --ckpt=.data/sheared-llama-2.7b.ckpt --tuning=lora
+ python3 script/9-profile.py --name='opt-2048' --tuning='lora' --module='mha' --backward
