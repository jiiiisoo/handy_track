import math
from typing import Mapping
import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from .network_builder import DictObsNetwork, DictObsBuilder


# * deprecated *
class SepDictObsNetwork(nn.Module):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.left_hand_net = DictObsNetwork(params, **kwargs)
        self.right_hand_net = DictObsNetwork(params, **kwargs)

    def is_rnn(self):
        return self.right_hand_net.is_rnn()

    def get_default_rnn_state(self):
        return self.right_hand_net.get_default_rnn_state()

    # def __getattribute__(self, name: str):
    #     # Access the attributes that belong to SepDictObsNetwork directly
    #     if name in ["left_hand_net", "right_hand_net", "load_state_dict", "forward", "__dict__"]:
    #         return object.__getattribute__(self, name)

    #     # For other attributes, delegate to left_hand_net
    #     return getattr(object.__getattribute__(self, "left_hand_net"), name)

    def forward(self, obs_dict):
        pass
        # * deprecated *
        assert False
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt_len = obs_dict.get("bptt_len", 0)

        obs = self.dict_feature_encoder(obs)

        if self.separate:
            a_out = c_out = obs
            a_out = self.actor_cnn(a_out)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            c_out = self.critic_cnn(c_out)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                if not self.is_rnn_before_mlp:
                    a_out_in = a_out
                    c_out_in = c_out
                    a_out = self.actor_mlp(a_out_in)
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = self.actor_cnn(out)
            out = out.flatten(1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.actor_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, out_in], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states


class SepDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = SepDictObsNetwork(self.params, **kwargs)
        return net
