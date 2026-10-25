# -*- coding: utf-8 -*-
"""机制定位与跨域表征相似性。

方法（参考文献）:
- 激活补丁/因果追踪(ROME 式): Thought Anchors (2506.19143)
- 注意力定位: Thought Anchors 的 "后续 token -> 目标 token" 注意力
- 表征相似性: CKA (Centered Kernel Alignment)

统一协议（领域无关, 长度对齐）:
  同一序列 = 前缀 + 目标句 T。
  - clean  : 正常前向, 记录逐层 hidden states, 参考延续 token R。
  - corrupt: 用 forward hook 把 T 的输入嵌入替换为整句均值(=机制层"删除内容")。
  - 补丁  : 用 forward hook 在 (layer l, 目标位置 p) 处把 corrupt 的 hidden state
            替换为 clean 的, 度量最后位置 logp(R) 的恢复程度 -> 逐层效应曲线 E(l)。

所有前向均走模型原生 forward(RoPE/位置编码由模型内部处理), 跨 transformers 版本健壮。
"""
import torch
import numpy as np


@torch.no_grad()
def forward_with_states(model, input_ids):
    """返回 (逐层后残差流列表, 最终 logits)。"""
    model.eval()
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    states = [hs for hs in out.hidden_states[1:]]  # 去掉嵌入层, 每层后各一个
    return states, out.logits


@torch.no_grad()
def forward_with_corrupted_embedding(model, input_ids, target_span):
    """用 hook 把目标 span 的输入嵌入替换为整句均值(机制层"删除内容")。"""
    model.eval()
    span_mean = None

    def emb_hook(module, args, output):
        nonlocal span_mean
        h = output.clone()
        if span_mean is None:
            span_mean = h[:, target_span[0]:target_span[1]].mean(dim=1, keepdim=True)
        h[:, target_span[0]:target_span[1]] = span_mean
        return h

    handle = model.model.embed_tokens.register_forward_hook(emb_hook)
    try:
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    states = [hs for hs in out.hidden_states[1:]]
    return states, out.logits


@torch.no_grad()
def forward_with_attention(model, input_ids):
    """返回注意力权重列表(每层 [batch, heads, q, kv])。"""
    model.eval()
    out = model(input_ids, output_attentions=True, use_cache=False)
    return out.attentions


@torch.no_grad()
def patched_logits(model, input_ids, clean_states, layer_idx, patch_pos):
    """用 hook 在 (layer_idx, patch_pos) 处把该层输出替换为 clean 的对应状态, 返回 logits。"""
    model.eval()

    def hook(module, args, output):
        # output 是张量 [batch, seq, hidden](transformers>=4.5x 层直接返回张量)
        h = output.clone()
        if h.dim() == 3:
            h[:, patch_pos, :] = clean_states[layer_idx][:, patch_pos, :]
        else:
            h[patch_pos, :] = clean_states[layer_idx][0, patch_pos, :]
        return h

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids)
    finally:
        handle.remove()
    return out.logits


def _logp(logits, pos, token_id):
    return float(torch.log_softmax(logits.float(), dim=-1)[0, pos, token_id])


def causal_trace(model, input_ids, target_pos, clean_states, corrupt_outcome, ref_token_id):
    """逐层激活补丁, 返回效应曲线。

    effect(l) = logp(ref | corrupt 补丁 layer l) - logp(ref | corrupt)
    """
    patch_pos = target_pos
    last = input_ids.shape[1] - 1
    num_layers = len(clean_states)
    effects = np.zeros(num_layers)
    for l in range(num_layers):
        logits = patched_logits(model, input_ids, clean_states, l, patch_pos)
        effects[l] = _logp(logits, last, ref_token_id) - corrupt_outcome
    return effects


@torch.no_grad()
def attention_weights(model, input_ids):
    """手动计算每层注意力权重(适配 Qwen2, 与 sdpa 兼容)。

    返回: list per layer, 每层 [num_heads, q_len, kv_len] (fp32)
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    model.eval()
    device = input_ids.device
    bsz, seq = input_ids.shape
    position_ids = torch.arange(0, seq, device=device).unsqueeze(0)
    hs = model.model.embed_tokens(input_ids)
    cos, sin = model.model.rotary_emb(hs, position_ids)
    attns = []
    for layer in model.model.layers:
        sa = layer.self_attn
        num_heads = sa.config.num_attention_heads
        num_kv = sa.config.num_key_value_heads
        residual = hs
        h = layer.input_layernorm(hs)
        q = sa.q_proj(h).view(bsz, seq, num_heads, sa.head_dim).transpose(1, 2)
        k = sa.k_proj(h).view(bsz, seq, num_kv, sa.head_dim).transpose(1, 2)
        v = sa.v_proj(h).view(bsz, seq, num_kv, sa.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = k.repeat_interleave(sa.num_key_value_groups, dim=1)
        v = v.repeat_interleave(sa.num_key_value_groups, dim=1)
        attn = (q.float() @ k.float().transpose(-1, -2)) * sa.scaling
        mask = torch.full((seq, seq), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attns.append(attn[0].half())  # fp16 节省显存
        # 继续前向(保持 hidden states 一致)
        attn_out = attn.half() @ v
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq, -1)
        hs = residual + sa.o_proj(attn_out)
        residual = hs
        hs = layer.mlp(layer.post_attention_layernorm(hs))
        hs = residual + hs
    return attns


def top_attention_heads(attentions, target_start, target_end, k=10):
    """后续 token -> 目标 token 平均注意力最高的 k 个 (score, layer, head)。"""
    scores = []
    for l, attn in enumerate(attentions):
        if attn is None:
            continue
        a = attn  # [heads, q, kv]
        q_after = a[:, target_end:, target_start:target_end]
        if q_after.numel() == 0:
            continue
        mean = q_after.mean(dim=(1, 2))
        for h in range(mean.shape[0]):
            scores.append((float(mean[h]), l, h))
    scores.sort(reverse=True)
    return scores[:k]


def cka(X, Y):
    """Centered Kernel Alignment (线性核)。X, Y: [n, d]"""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    denom = np.sqrt((K * K).sum() * (L * L).sum())
    return float((K * L).sum() / denom) if denom > 0 else 0.0


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
