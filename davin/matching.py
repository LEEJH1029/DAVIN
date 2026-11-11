"""Multi-granular matching.

  S_cg(d, i) = cos(h_d, h_i)
  S_fg(d, i) = Σ_n Σ_m α_nm · cos(h_{p_n}, h_{a_m})
  S(d, i)    = λ · S_cg(d, i) + (1 − λ) · S_fg(d, i),   λ = σ(θ) learned
"""
import math

import torch
from torch import nn


class MultiGranularMatching(nn.Module):
    def __init__(self, joint_attention, lambda_init=.5):
        super().__init__()
        if not 0. < lambda_init < 1.:
            raise ValueError('lambda_init must be in (0, 1)')
        self.joint_attention = joint_attention
        self.theta = nn.Parameter(torch.tensor(math.log(lambda_init / (1. - lambda_init)), dtype=torch.float32,
                                               device=joint_attention.W_p.weight.device))

    @property
    def lam(self):
        return self.theta.sigmoid()

    def forward(self, h_d, preferences, items, n_dialogues, n_items):
        """S of shape [n_dialogues, n_items]."""
        pref_emb, pref_owner = preferences
        h_i, attr_emb, attr_owner = items
        s_cg = h_d @ h_i.t()
        s_fg = self.joint_attention.fine_grained_score(pref_emb, attr_emb, attr_owner, n_items, pref_owner, n_dialogues)
        lam = self.lam
        return lam * s_cg + (1 - lam) * s_fg
