import math
import sys
import re
from typing import Optional, List
from collections import defaultdict

import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import LayerNorm, SamePad
from fairseq.utils import index_put

DBG=True if len(sys.argv) == 1 or re.match(r"(.*)(prune.py|save_final_ckpt.py|merge.py)", sys.argv[0]) else False

if DBG:
    from hardconcrete import HardConcrete
    from pruning_utils import prune_linear_layer
else:
    from .hardconcrete import HardConcrete
    from .pruning_utils import prune_linear_layer


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )

        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                EncoderLayer(          
                    embedding_dim=self.embedding_dim,
                    use_ffn=args.encoder_use_ffn[i] if args.encoder_use_ffn else True,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim_detailed[i] if args.encoder_ffn_embed_dim_detailed else args.encoder_ffn_embed_dim,
                    use_attention=args.encoder_use_attention[i] if args.encoder_use_attention else True, 
                    num_attention_heads=args.encoder_attention_heads_detailed[i] if args.encoder_attention_heads_detailed else args.encoder_attention_heads,
                    attention_heads_dim=args.encoder_attention_heads_dim,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    prune_attention_heads=args.encoder_prune_attention_heads,
                    prune_attention_layer=args.encoder_prune_attention_layer,
                    prune_ffn_intermediate=args.encoder_prune_ffn_intermediate,
                    prune_ffn_layer=args.encoder_prune_ffn_layer,
                    merge_type=args.encoder_merge_type,
                )
                for i in range(args.encoder_layers)
            ]
        )
        
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results
    
    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        return x, layer_results
    
    def get_intermediate_outputs(self, x, padding_mask=None):
        ret: List[Tensor] = []
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)
        ret.append(x)
        
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            ret.append(x)

        return ret
    
    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
    
    def merge(self, t=0.01, type="QKV"):
        for layer in self.layers:
            layer.merge(t, type)
    
    def prune(self):
        new_config = defaultdict(list)
        for layer in self.layers:
            layer_config = layer.prune()
            new_config["encoder_use_attention"].append(layer_config["encoder_use_attention"])
            new_config["encoder_attention_heads"].append(layer_config["encoder_attention_heads"])
            new_config["encoder_use_ffn"].append(layer_config["encoder_use_ffn"])
            new_config["encoder_ffn_embed_dim"].append(layer_config["encoder_ffn_embed_dim"])
        return new_config
    
    def get_num_params(self):
        num_params = sum(p.numel() for p in self.pos_conv.parameters()) # pos_conv
        num_params += self.embedding_dim * 2 # layer norm
        for layer in self.layers:
            num_params += layer.get_num_params()
        return num_params
    
    
class EncoderLayer(nn.Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""
    def __init__(
        self,       
        embedding_dim: float = 768,
        use_ffn: bool = True,
        ffn_embedding_dim: float = 3072,
        use_attention: bool = True,
        num_attention_heads: float = 8,
        attention_heads_dim: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        prune_attention_heads: bool = False,
        prune_attention_layer: bool = False,
        prune_ffn_intermediate: bool = False,
        prune_ffn_layer: bool = False,
        merge_type: str = None,
    ):
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn) if use_ffn else None
        self.self_attn = SelfAttention(
            self.embedding_dim,
            num_attention_heads,
            attention_heads_dim,
            dropout=attention_dropout,
            prune_attention_heads=prune_attention_heads,
            prune_attention_layer=prune_attention_layer,
            merge_type=merge_type
        ) if use_attention else None
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout) if use_ffn else None
        self.dropout3 = nn.Dropout(dropout) if use_ffn else None

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim) if use_ffn else None
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim) if use_ffn else None

        # layer norm associated with the position wise FNN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        if prune_ffn_intermediate:
            self.hard_concrete_for_intermediate = HardConcrete(
                n_in=ffn_embedding_dim, init_mean=0.5
            )
        else:
            self.hard_concrete_for_intermediate = None
        
        if prune_ffn_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x: Tensor, self_attn_padding_mask: Tensor = None) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(batch, frame, feature)``.
            self_attn_padding_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
        Returns:
            x: Shapes are the same as in the input.
        """
        attn = None
        if self.self_attn is not None:
            residual = x
            if self.layer_norm_first:
                x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(x, attention_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x

        if self.layer_norm_first:
            if self.fc1 is not None:
                residual = x
                x = self.final_layer_norm(x)
                x = residual + self.forward_ffn(x)
        else:
            # NOTE: for post norm, the layer norms should always be applied even if the layers are pruned.
            x = self.self_attn_layer_norm(x)
            if self.fc1 is not None:
                x = x + self.forward_ffn(x)
            x = self.final_layer_norm(x)
        return x, attn
    
    def forward_ffn(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        if self.hard_concrete_for_intermediate is not None:
            intermediate_mask = self.hard_concrete_for_intermediate()   # (intermediate_features,)
            x = x * intermediate_mask
        x = self.fc2(x)
        x = self.dropout3(x)
        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer()     # (1,)
            x = x * layer_mask
        return x
    
    def merge(self, t=0.01, type="QKV"):
        if self.self_attn is not None:
            self.self_attn.merge(t, type)
    
    def prune(self):
        new_config = {
            "encoder_use_ffn": True,
            "encoder_ffn_embed_dim": self.fc1.out_features
        }
        self_attn_config = self.self_attn.prune()
        new_config.update(self_attn_config)
        if not self_attn_config["encoder_use_attention"]:
            self.self_attn = None
            
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.fc2.weight.data *= layer_mask
            self.fc2.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["encoder_use_ffn"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_intermediate is not None:
            assert not self.hard_concrete_for_intermediate.training
            interm_mask = self.hard_concrete_for_intermediate()
            interm_index = interm_mask.nonzero().squeeze(-1)    # NOTE: must specify dim=-1
            new_config["encoder_ffn_embed_dim"] = len(interm_index)
            if new_config["encoder_ffn_embed_dim"] == 0:
                new_config["encoder_use_ffn"] = False
            else:
                prune_linear_layer(self.fc1, interm_index, "output")
                self.fc2.weight.data *= interm_mask
                prune_linear_layer(self.fc2, interm_index, "input")
            self.hard_concrete_for_intermediate = None
        if not new_config["encoder_use_ffn"]:
            self.fc1 = None
            self.activation_fn = None
            self.dropout2 = None
            self.fc2 = None
            self.dropout3 = None
        
        return new_config
    
    def get_num_params(self):
        num_params = self.embedding_dim * 2 * 2     # two layer norms
        if self.self_attn is not None:
            num_params += self.self_attn.get_num_params()
        if self.fc1 is not None:
            if self.hard_concrete_for_intermediate is not None:
                intermediate_features = self.hard_concrete_for_intermediate.l0_norm()
            else:
                intermediate_features = self.fc1.out_features
            ffn_num_params = (self.fc1.in_features + 1) * intermediate_features
            ffn_num_params += (intermediate_features + 1) * self.fc2.out_features
            if self.hard_concrete_for_layer is not None:
                ffn_num_params *= self.hard_concrete_for_layer.l0_norm()
            num_params += ffn_num_params
        return num_params
    
    
class SelfAttention(nn.Module):
    """Multihead Self Attention module

    Args:
        cfg (TransformerEncoderConfig): Transformer Encoder config.
    """

    def __init__(
            self,
            embed_dim: float = 768,
            num_heads: float = 8,
            head_dim: float = 64,
            dropout: float = 0.0,
            prune_attention_heads: bool = False,
            prune_attention_layer: bool = False,
            merge_type: str = None,

        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = torch.nn.Dropout(dropout)

        self.scaling = self.head_dim**-0.5

        self.merge_type = merge_type

        if self.merge_type is not None:
            assert self.merge_type in ["QKV", "QK", "QV", "KV"]
            self.h_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
            if self.merge_type == "QK":
                self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
            elif self.merge_type == "QV":
                self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
            elif self.merge_type == "KV":
                self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        else:
            self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
            self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

        if prune_attention_heads:
            self.hard_concrete_for_heads = HardConcrete(n_in=self.num_heads, init_mean=0.01)
        else:
            self.hard_concrete_for_heads = None

        if prune_attention_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape: ``(batch, frame, feature)``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
        Returns:
            Tensor: The resulting attention output. Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )

        batch_size, length, embed_dim = x.size()
        shape = (batch_size, length, self.num_heads, self.head_dim)

        if self.merge_type is not None:
            h = self.h_proj(x).view(*shape)
            q = self.q_proj(x).view(*shape).transpose(2, 1)  if self.merge_type == "KV" else h.transpose(2, 1)
            k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1) if self.merge_type == "QV" else h.permute(0, 2, 3, 1)
            v = self.v_proj(x).view(*shape).transpose(2, 1) if self.merge_type == "QK" else h.transpose(2, 1) if self.merge_type == "KV" else q
        else:
            q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
            k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
            v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        weights = (self.scaling * q) @ k  # B, nH, L, L

        # don't attend to padding symbols
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),float("-inf"))
            
        # subtracting a constant value from the tensor won't change the output of softmax.
        # apply the subtraction to avoid value overflow in torch.nn.functional.softmax.
        # for more details, please see Equation 7 in https://arxiv.org/abs/2112.08778
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd

        if self.hard_concrete_for_heads is not None:
            head_mask = self.hard_concrete_for_heads()  # (nH,)
            output = output * head_mask.unsqueeze(-1).unsqueeze(-1)

        output = output.transpose(2, 1).reshape(batch_size, length, self.num_heads * self.head_dim)
        output = self.out_proj(output)

        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer() # (1,)
            output = output * layer_mask

        return output, weights
    
    def merge(self, t=0.01, type="QKV"):
        assert type in ["QKV", "QK", "QV", "KV"]
        merge_proj = []
        if type != "KV":
            merge_proj.append(self.q_proj)
        if type != "QK":
            merge_proj.append(self.v_proj)
        if type != "QV":
            merge_proj.append(self.k_proj)

        weights = [p.weight.clone().detach() for p in merge_proj]
        biases = [p.bias.clone().detach() for p in merge_proj]

        avg_weight = torch.mean(torch.stack(weights), dim=0)
        avg_bias = torch.mean(torch.stack(biases), dim=0)

        weights = [w - t * (w - avg_weight) for w in weights]
        biases = [b - t * (b - avg_bias) for b in biases]

        for p, w, b in zip(merge_proj, weights, biases):
            p.weight = nn.Parameter(w)
            p.bias = nn.Parameter(b)
    
    def prune(self):
        new_config = {
            "encoder_use_attention": True,
            "encoder_attention_heads": self.num_heads,
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["encoder_use_attention"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["encoder_attention_heads"] = len(head_mask.nonzero())
            if new_config["encoder_attention_heads"] == 0:
                new_config["encoder_use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None
            self.num_heads = new_config["encoder_attention_heads"]

        return new_config
    
    def get_num_params(self):
        if self.merge_type is None:
            num_weights = 3
        elif self.merge_type == "QKV":
            num_weights = 1
        else:
            num_weights = 2
        if self.hard_concrete_for_heads is not None:
            num_heads = self.hard_concrete_for_heads.l0_norm()
        else:
            num_heads = self.num_heads
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * num_weights \
            + (num_heads * self.head_dim + 1) * self.embed_dim
        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        return num_params
    

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for SelfAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, SelfAttention) and hasattr(module, "q_proj"):
        normal_(module.q_proj.weight.data)
    if isinstance(module, SelfAttention) and hasattr(module, "k_proj"):
        normal_(module.k_proj.weight.data)
    if isinstance(module, SelfAttention) and hasattr(module, "v_proj"):
        normal_(module.v_proj.weight.data)
