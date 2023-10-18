import os
import torch
import argparse
import transformers as T
from naive_gpt import models


class OPTModel(models.OPTModel):
    def load_pretrained(self, state_dict: dict):
        # embeddings
        self.embedding.load_state_dict({
            'weight': state_dict.pop(
                'model.decoder.embed_tokens.weight'
            )
        })
        self.learned_pe.load_state_dict({
            'weight': state_dict.pop(
                'model.decoder.embed_positions.weight'
            )
        })

        # decoder layers
        for i, v in enumerate(self.decoders):
            prefix = 'model.decoder.layers.{}.'.format(i)

            # dump states
            layer_state = {}
            removed_keys = []
            for k in state_dict:
                k: str
                if not k.startswith(prefix):
                    continue
                new_k = k.removeprefix(prefix)
                layer_state[new_k] = state_dict[k]
                removed_keys.append(k)
            for k in removed_keys:
                state_dict.pop(k)

            # feedforwards
            v.load_state_dict({
                # ffd
                'ffd.fc1.bias': layer_state.pop('fc1.bias'),
                'ffd.fc1.weight': layer_state.pop('fc1.weight'),
                'ffd.fc2.bias': layer_state.pop('fc2.bias'),
                'ffd.fc2.weight': layer_state.pop('fc2.weight'),
                # mha
                'mha.linear_q.bias': layer_state.pop('self_attn.q_proj.bias'),
                'mha.linear_q.weight': layer_state.pop('self_attn.q_proj.weight'),
                'mha.linear_k.bias': layer_state.pop('self_attn.k_proj.bias'),
                'mha.linear_k.weight': layer_state.pop('self_attn.k_proj.weight'),
                'mha.linear_v.bias': layer_state.pop('self_attn.v_proj.bias'),
                'mha.linear_v.weight': layer_state.pop('self_attn.v_proj.weight'),
                'mha.linear_o.bias': layer_state.pop('self_attn.out_proj.bias'),
                'mha.linear_o.weight': layer_state.pop('self_attn.out_proj.weight'),
                # norms
                'norm1.bias': layer_state.pop('self_attn_layer_norm.bias'),
                'norm1.weight': layer_state.pop('self_attn_layer_norm.weight'),
                'norm2.bias': layer_state.pop('final_layer_norm.bias'),
                'norm2.weight': layer_state.pop('final_layer_norm.weight'),
            })
            assert len(layer_state) == 0

        # final norm and output
        self.final_norm.load_state_dict({
            'bias': state_dict.pop(
                'model.decoder.final_layer_norm.bias'
            ),
            'weight': state_dict.pop(
                'model.decoder.final_layer_norm.weight'
            )
        })
        self.lm_output.load_state_dict({
            'weight': state_dict.pop('lm_head.weight')
        })

        # check
        assert len(state_dict) == 0


class LLaMAModel(models.LLaMAModel):
    def load_pretrained(self, state_dict: dict):
        # embedding
        self.embedding.load_state_dict({
            'weight': state_dict.pop(
                'model.embed_tokens.weight'
            )
        })

        # decoder layers
        for i, v in enumerate(self.decoders):
            prefix = 'model.layers.{}.'.format(i)

            # dump states
            layer_state = {}
            removed_keys = []
            for k in state_dict:
                k: str
                if not k.startswith(prefix):
                    continue
                new_k = k.removeprefix(prefix)
                layer_state[new_k] = state_dict[k]
                removed_keys.append(k)
            for k in removed_keys:
                state_dict.pop(k)

            #
            missing_keys = v.load_state_dict({
                # ffd
                'ffd.side.weight': layer_state.pop('mlp.up_proj.weight'),
                'ffd.gate.weight': layer_state.pop('mlp.gate_proj.weight'),
                'ffd.down.weight': layer_state.pop('mlp.down_proj.weight'),
                # mha
                'mha.linear_q.weight': layer_state.pop('self_attn.q_proj.weight'),
                'mha.linear_k.weight': layer_state.pop('self_attn.k_proj.weight'),
                'mha.linear_v.weight': layer_state.pop('self_attn.v_proj.weight'),
                'mha.linear_o.weight': layer_state.pop('self_attn.o_proj.weight'),
                # norms
                'norm1.weight': layer_state.pop('input_layernorm.weight'),
                'norm2.weight': layer_state.pop('post_attention_layernorm.weight')
            }, strict=False).missing_keys
            assert len(missing_keys) == 3
            assert len(layer_state) == 1

        # final norm and output
        self.final_norm.load_state_dict({
            'weight': state_dict.pop('model.norm.weight')
        })
        self.lm_output.load_state_dict({
            'weight': state_dict.pop('lm_head.weight')
        })

        # check
        assert len(state_dict) == 0


def convert(name: str):
    # load model
    if name.lower().find('opt') != -1:
        ModelType = OPTModel
        PretrainedType = T.OPTForCausalLM
    elif name.lower().find('llama') != -1:
        ModelType = LLaMAModel
        PretrainedType = T.LlamaForCausalLM
    else:
        raise NotImplementedError
    pretrained = PretrainedType.from_pretrained(name)
    state_dict = pretrained.state_dict()
    pretrained.eval()

    # convert model
    def get_feedforward_dim():
        if hasattr(pretrained.config, 'ffn_dim'):
            return pretrained.config.ffn_dim
        return pretrained.config.intermediate_size
    config = {
        'd_model': pretrained.config.hidden_size,
        'n_heads': pretrained.config.num_attention_heads,
        'n_layers': pretrained.config.num_hidden_layers,
        'vocab_size': pretrained.config.vocab_size,
        'd_feedforward': get_feedforward_dim(),
        'max_length': pretrained.config.max_position_embeddings,
        'p_dropout': 0.0
    }
    model = ModelType(**config)
    model.load_pretrained(state_dict)
    model.eval()

    # check output
    batch_size = 16
    seq_length = 64
    x = torch.randint(
        high=config['vocab_size'],
        size=[batch_size, seq_length]
    )
    y_1, y_2 = pretrained(x)['logits'], model(x)
    if name.lower().find('sheared'):
        # a small different in sheared-llama
        # "rms_norm_eps": 1e-05
        assert torch.abs(y_1 - y_2).mean() < 0.1
    else:
        assert torch.allclose(y_1, y_2, atol=1e-3)

    # dump model
    try:
        os.mkdir('.data')
    except FileExistsError:
        pass
    name = name.lower().split('/')[-1]
    torch.save(
        {
            'config': config,
            'state_dict': model.state_dict()
        },
        f='.data/{}.ckpt'.format(name)
    )

    #
    print('dump {} finish'.format(name))


def main():
    # facebook/opt-125m
    # facebook/opt-1.3b
    # facebook/opt-2.7b
    # princeton-nlp/Sheared-LLaMA-2.7B
    # openlm-research/open_llama_7b
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default='facebook/opt-1.3b',
        help='specify model name or path'
    )
    args = parser.parse_args()
    convert(name=args.name)


if __name__ == '__main__':
    main()
