from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class BoundedNormal:
    """Diagonal Normal mapped into finite action bounds with tanh."""

    def __init__(
        self,
        location: torch.Tensor,
        scale: torch.Tensor,
        *,
        low: float | tuple[float, ...] = -1.0,
        high: float | tuple[float, ...] = 1.0,
    ) -> None:
        if not location.is_floating_point() or not scale.is_floating_point():
            raise ValueError("Normal parameters must be floating-point tensors")
        if bool(torch.any(~torch.isfinite(location))):
            raise ValueError("Normal location must be finite")
        if bool(torch.any(~torch.isfinite(scale) | (scale <= 0.0))):
            raise ValueError("Normal scale must be finite and positive")
        self.base = torch.distributions.Normal(location, scale)
        self.low = torch.as_tensor(
            low,
            dtype=self.base.loc.dtype,
            device=self.base.loc.device,
        )
        self.high = torch.as_tensor(
            high,
            dtype=self.base.loc.dtype,
            device=self.base.loc.device,
        )
        try:
            self.low = torch.broadcast_to(self.low, self.base.loc.shape)
            self.high = torch.broadcast_to(self.high, self.base.loc.shape)
        except RuntimeError as exc:
            raise ValueError("action bounds do not match Normal parameters") from exc
        if bool(torch.any(~torch.isfinite(self.low) | ~torch.isfinite(self.high))):
            raise ValueError("action bounds must be finite")
        if bool(torch.any(self.high <= self.low)):
            raise ValueError("each action upper bound must exceed its lower bound")
        self._midpoint = 0.5 * (self.high + self.low)
        self._half_range = 0.5 * (self.high - self.low)

    @classmethod
    def from_mode(
        cls,
        mode: torch.Tensor,
        scale: torch.Tensor,
        *,
        low: float | tuple[float, ...] = -1.0,
        high: float | tuple[float, ...] = 1.0,
    ) -> BoundedNormal:
        low_tensor = torch.as_tensor(low, dtype=mode.dtype, device=mode.device)
        high_tensor = torch.as_tensor(
            high,
            dtype=mode.dtype,
            device=mode.device,
        )
        if bool(torch.any(high_tensor <= low_tensor)):
            raise ValueError("each action upper bound must exceed its lower bound")
        if bool(torch.any((mode < low_tensor) | (mode > high_tensor))):
            raise ValueError("mode is outside the requested action bounds")
        midpoint = 0.5 * (high_tensor + low_tensor)
        half_range = 0.5 * (high_tensor - low_tensor)
        normalized = (mode - midpoint) / half_range
        epsilon = torch.finfo(mode.dtype).eps
        location = torch.atanh(
            normalized.clamp(-1.0 + epsilon, 1.0 - epsilon)
        )
        return cls(location, scale, low=low, high=high)

    @property
    def mode(self) -> torch.Tensor:
        return self._transform(self.base.loc)

    def sample(self) -> torch.Tensor:
        return self._transform(self.base.sample())

    def rsample(self) -> torch.Tensor:
        return self._transform(self.base.rsample())

    def sample_with_pre_tanh(self) -> tuple[torch.Tensor, torch.Tensor]:
        pre_tanh = self.base.sample()
        return self._transform(pre_tanh), pre_tanh

    def rsample_with_pre_tanh(self) -> tuple[torch.Tensor, torch.Tensor]:
        pre_tanh = self.base.rsample()
        return self._transform(pre_tanh), pre_tanh

    def sample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        action, pre_tanh = self.sample_with_pre_tanh()
        return action, self.log_prob_from_pre_tanh(pre_tanh)

    def rsample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        action, pre_tanh = self.rsample_with_pre_tanh()
        return action, self.log_prob_from_pre_tanh(pre_tanh)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if bool(torch.any((action < self.low) | (action > self.high))):
            raise ValueError("action is outside the bounded distribution support")
        return self.log_prob_from_pre_tanh(self._inverse(action))

    def log_prob_from_pre_tanh(
        self,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        elementwise = self.base.log_prob(pre_tanh)
        elementwise -= self._log_abs_det_jacobian(pre_tanh)
        return elementwise.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        pre_tanh = self.base.rsample()
        return (
            self.base.entropy() + self._log_abs_det_jacobian(pre_tanh)
        ).sum(dim=-1)

    def _transform(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(pre_tanh.dtype).eps
        normalized = torch.tanh(pre_tanh).clamp(
            min=-1.0 + epsilon,
            max=1.0 - epsilon,
        )
        return self._midpoint + self._half_range * normalized

    def _inverse(self, action: torch.Tensor) -> torch.Tensor:
        normalized = (action - self._midpoint) / self._half_range
        epsilon = torch.finfo(action.dtype).eps
        normalized = normalized.clamp(
            min=-1.0 + epsilon,
            max=1.0 - epsilon,
        )
        return torch.atanh(normalized)

    def _log_abs_det_jacobian(
        self,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        tanh_log_jacobian = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return torch.log(self._half_range) + tanh_log_jacobian
