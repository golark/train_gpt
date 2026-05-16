from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # caching of rotary freq buffers (optional)
        self.register_buffer("_rope_freqs", torch.empty(0), persistent=False)

    @staticmethod
    def _rotate_half(x):
        # x: (..., dim)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # (-x2, x1) interleaved
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x_rot

    @staticmethod
    def _make_rotary_emb(seq_len, dim, device, dtype):
        # dim must be even
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum('i,j->ij', positions, inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        # q,k: (B, nh, T, hs)
        # cos/sin: (1,1,T,hs) or broadcastable
        q_out = q * cos + CausalSelfAttention._rotate_half(q) * sin
        k_out = k * cos + CausalSelfAttention._rotate_half(k) * sin
        return q_out, k_out

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape for heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- APPLY ROTARY POSITIONAL EMBEDDINGS TO q, k ---
        # compute cos/sin for this sequence length & head_dim
        # use dtype/device of q
        dtype = q.dtype
        device = q.device
        # ensure head_dim is even for interleaving
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        cos, sin = self._make_rotary_emb(T, self.head_dim, device=device, dtype=dtype)
        # cos,sin shape (1,1,T,hs)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # attention with scaled dot-product (uses flash attention when available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # NOTE: with RoPE we DO NOT use absolute learned position embeddings.
        # We only keep token embeddings and apply RoPE inside attention.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def to_hf(self, save_directory):
        """
        Export this RoPE minGPT model to a HuggingFace GPT2LMHeadModel and save it to `save_directory`.

        This implementation is robust to the fact that RoPE models don't use learned position
        embeddings (so HF's `transformer.wpe` will be left as the HF default), and copies
        any matching parameters by name. It also transposes Conv1D-like weights when needed.
        """
        import os
        from transformers import GPT2LMHeadModel, GPT2Config

        # build HF config from this model's config
        cfg_hf = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=getattr(self.config, 'block_size', getattr(self.config, 'n_positions', 1024)),
            n_ctx=getattr(self.config, 'block_size', getattr(self.config, 'n_ctx', 1024)),
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
        )
        model_hf = GPT2LMHeadModel(cfg_hf)

        # source (this model) and target (HF) state dicts
        sd_src = self.state_dict()
        # filter out any buffers that are not parameters (mirrors from_pretrained logic)
        sd_src = {k: v for k, v in sd_src.items() if not k.endswith('.attn.bias')}

        sd_tgt = model_hf.state_dict()
        sd_tgt = {k: v for k, v in sd_tgt.items() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')}

        # helper: list of suffixes that require transposition (Conv1D -> Linear)
        transposed_suffixes = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight',
                               'c_attn.weight', 'c_proj.weight', 'c_fc.weight', 'c_proj.weight')

        # iterate over HF target keys and copy from source when a name match exists
        copied = []
        skipped = []
        mismatches = []
        for k_tgt in sd_tgt.keys():
            if k_tgt in sd_src:
                src_param = sd_src[k_tgt]
                tgt_param = sd_tgt[k_tgt]
                try:
                    if any(k_tgt.endswith(suf) for suf in transposed_suffixes):
                        # transpose Conv1D-like weight if shapes indicate
                        if src_param.shape[::-1] == tgt_param.shape:
                            with torch.no_grad():
                                sd_tgt[k_tgt].copy_(src_param.t())
                            copied.append(k_tgt)
                        elif src_param.shape == tgt_param.shape:
                            with torch.no_grad():
                                sd_tgt[k_tgt].copy_(src_param)
                            copied.append(k_tgt)
                        else:
                            mismatches.append((k_tgt, src_param.shape, tgt_param.shape))
                    else:
                        if src_param.shape == tgt_param.shape:
                            with torch.no_grad():
                                sd_tgt[k_tgt].copy_(src_param)
                            copied.append(k_tgt)
                        else:
                            mismatches.append((k_tgt, src_param.shape, tgt_param.shape))
                except Exception as e:
                    mismatches.append((k_tgt, str(e)))
            else:
                # no corresponding parameter in the RoPE model; this commonly happens for
                # HF position embeddings (`transformer.wpe`) which RoPE models don't use.
                skipped.append(k_tgt)

        # load the modified target state dict into the HF model
        model_hf.load_state_dict(sd_tgt)

        # ensure output directory exists and save
        os.makedirs(save_directory, exist_ok=True)
        model_hf.save_pretrained(save_directory)

        # return a small report
        report = {
            'copied': len(copied),
            'skipped': len(skipped),
            'mismatches': mismatches[:20],
        }
        print(f"to_hf: copied={report['copied']} skipped={report['skipped']} mismatches={len(report['mismatches'])}")
        return report

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token embeddings (no learned position embeddings with RoPE)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # from_pretrained and other methods omitted for brevity; can be re-used with minimal edits

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=False):

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
