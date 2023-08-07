import os
import torch
import argparse
import transformers as T
from naive_gpt import models


class OPTModel(models.OPTModel):
    def load_pretrained(self, state_dict: dict):
        # embeddings
        self.embedding.load_state_dict({
            '0.weight': state_dict.pop(
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
                'ffd.fc.0.bias': layer_state.pop('fc1.bias'),
                'ffd.fc.0.weight': layer_state.pop('fc1.weight'),
                'ffd.fc.3.bias': layer_state.pop('fc2.bias'),
                'ffd.fc.3.weight': layer_state.pop('fc2.weight'),
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

        # cleanup
        assert len(state_dict) == 0


def convert(name: str):
    # load model
    OptModel = T.OPTForCausalLM
    opt = OptModel.from_pretrained(name)
    state_dict = opt.state_dict()
    opt.eval()

    # convert model
    config = opt.config
    model = OPTModel(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        p_dropout=config.dropout,
        vocab_size=config.vocab_size,
        d_embedding=config.word_embed_proj_dim,
        d_feedforward=config.ffn_dim,
        max_length=config.max_position_embeddings
    )
    model.load_pretrained(state_dict)
    model.eval()

    # check output
    batch_size = 16
    seq_length = 64
    x = torch.randint(
        high=config.vocab_size,
        size=[batch_size, seq_length]
    )
    y_1, y_2 = opt(x)['logits'], model(x)
    assert torch.allclose(
        y_1, y_2, atol=1e-5, rtol=1e-3
    )

    # dump model
    try:
        os.mkdir('.data')
    except FileExistsError:
        pass
    name = name.split('/')[-1]
    torch.save(
        model.state_dict(),
        f='.data/{}.ckpt'.format(name)
    )

    #
    print('dump {} finish'.format(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', help='specify model name or path',
        default='facebook/opt-125m'
    )
    args = parser.parse_args()
    convert(name=args.name)


if __name__ == '__main__':
    main()
