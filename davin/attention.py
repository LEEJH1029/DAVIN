"""Joint attention component: the degree of relevance α_nm of each preference-attribute pair.

  Q = H_P W_p,   Z = H_A W_a,   R = Q Zᵀ / √d_k
  α_nm       = exp(r_nm) / Σ_n' Σ_m' exp(r_n'm')        over all pairs of a (dialogue, item)
  S_fg(d, i) = Σ_n Σ_m α_nm · cos(h_{p_n}, h_{a_m})

Preferences and attributes are flat matrices; owner vectors map each row to its dialogue or item.
"""
import math

import torch
from torch import nn


def joint_softmax(logits, pref_owner, attr_owner, n_dialogues, n_items):
    """Softmax over all (preference, attribute) pairs of each (dialogue, item)."""
    p = logits.size(0)
    attr_index = attr_owner[None, :].expand(p, -1)
    pref_index = pref_owner[:, None].expand(-1, n_items)
    with torch.no_grad():
        row_max = logits.new_full((p, n_items), -torch.inf).scatter_reduce(
            1, attr_index, logits, reduce='amax', include_self=True)
        block_max = logits.new_full((n_dialogues, n_items), -torch.inf).scatter_reduce(
            0, pref_index, row_max, reduce='amax', include_self=True)
        block_max = torch.where(torch.isfinite(block_max), block_max, 0.)
        shift = block_max[pref_owner].gather(1, attr_index)
    numerator = (logits - shift).exp()
    row_sum = logits.new_zeros(p, n_items).scatter_add(1, attr_index, numerator)
    denominator = logits.new_zeros(n_dialogues, n_items).index_add(0, pref_owner, row_sum)
    return numerator / denominator[pref_owner].gather(1, attr_index).clamp_min(1e-30)


class JointAttention(nn.Module):
    def __init__(self, d_h, d_k=256):
        super().__init__()
        self.d_h, self.d_k = int(d_h), int(d_k)
        self.W_p = nn.Linear(self.d_h, self.d_k, bias=False)
        self.W_a = nn.Linear(self.d_h, self.d_k, bias=False)
        nn.init.zeros_(self.W_p.weight)       # Training starts from equal relevance for all pairs.
        nn.init.xavier_uniform_(self.W_a.weight)

    def get_config(self):
        return dict(d_h=self.d_h, d_k=self.d_k)

    def fine_grained_score(self, pref_emb, attr_emb, attr_owner, n_items, pref_owner, n_dialogues):
        """S_fg of each (dialogue, item). Embeddings must be L2-normalized."""
        pref, attrs = pref_emb.float(), attr_emb.float()
        p = len(pref)
        if not p or not len(attrs):
            return pref.new_zeros(n_dialogues, n_items) + 0. * (pref.sum() + attrs.sum())
        r = self.W_p(pref) @ self.W_a(attrs).T / math.sqrt(self.d_k)
        alpha = joint_softmax(r, pref_owner, attr_owner, n_dialogues, n_items)
        cosine = pref @ attrs.T
        contributions = cosine.new_zeros(p, n_items).scatter_add(1, attr_owner[None, :].expand(p, -1), alpha * cosine)
        return cosine.new_zeros(n_dialogues, n_items).index_add(0, pref_owner, contributions)
