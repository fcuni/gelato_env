from typing import Optional
import torch
from einops import reduce
from torch import einsum
from torch.distributions import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, probs: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        probs_or_logits = probs if probs is not None else logits
        self.mask = mask
        probs_or_logits = torch.atleast_2d(probs_or_logits)
        self.batch, self.nb_action = probs_or_logits.size()
        if mask is None:
            if logits is not None:
                super(CategoricalMasked, self).__init__(logits=probs_or_logits, validate_args=False)
            else:
                super(CategoricalMasked, self).__init__(probs=probs_or_logits, validate_args=False)
        else:
            self.mask = mask.bool()
            self.mask_value = torch.tensor(torch.finfo(probs_or_logits.dtype).min, dtype=probs_or_logits.dtype) \
                if logits is not None else torch.tensor(0, dtype=probs_or_logits.dtype)
            probs_or_logits = torch.where(self.mask, probs_or_logits, self.mask_value)
            if logits is not None:
                super(CategoricalMasked, self).__init__(logits=probs_or_logits, validate_args=False)
            else:
                super(CategoricalMasked, self).__init__(probs=probs_or_logits, validate_args=False)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
