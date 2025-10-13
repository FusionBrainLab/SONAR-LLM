import torch
from typing import Dict, List, Optional


class PenaltySampler:
    # Whether to update the base vector or the noise vector
    to_base_vector = False
    to_noise_vector = False


    def __init__(self, repetition_penalty: float = 0.0, repetition_memory: int = 0, cfg: Optional[Dict] = None) -> None:
        self.repetition_penalty = float(repetition_penalty) if repetition_penalty is not None else 0.0
        self.repetition_memory = int(repetition_memory) if repetition_memory is not None else 0
        self.to_base_vector = getattr(cfg, "to_base_vector", False) if cfg is not None else False
        self.to_noise_vector = getattr(cfg, "to_noise_vector", False) if cfg is not None else False
    
    def _sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        pass

    def _update_base_vector(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
    ) -> torch.Tensor:
        pass

    def sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.to_noise_vector:
            return self._sample_unit_direction(final_hidden, recent_dirs, rng=rng)
        else:
            return torch.randn_like(final_hidden) if rng is None else torch.randn_like(final_hidden, generator=rng)

    def update_base_vector(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
    ) -> torch.Tensor:
        if self.to_base_vector:
            return self._update_base_vector(final_hidden, recent_dirs)
        else:
            return final_hidden

    def update_recent_dirs(
        self,
        recent: List[torch.Tensor],
        u: torch.Tensor,
    ) -> List[torch.Tensor]:
        memory_cap = int(self.repetition_memory) if self.repetition_memory is not None else 0
        if memory_cap <= 0:
            return recent
        if not torch.isfinite(u).all():
            return recent
        new_recent = recent + [u.detach().to("cpu")]
        if len(new_recent) > memory_cap:
            new_recent = new_recent[-memory_cap:]
        return new_recent


class DirectionWithMeanShiftPenalty(PenaltySampler):
    def __init__(self, repetition_penalty: float = 0.0, repetition_memory: int = 0, cfg: Optional[Dict] = None) -> None:
        super().__init__(repetition_penalty=repetition_penalty, repetition_memory=repetition_memory, cfg=cfg)

    @staticmethod
    def _apply_repetition_penalty(
        g: torch.Tensor,
        penalty: float,
        memory_cap: int,
        recent_dirs: List[torch.Tensor],
    ) -> torch.Tensor:
        if memory_cap <= 0 or len(recent_dirs) == 0:
            return g
        B = torch.stack([u.to(device=g.device, dtype=g.dtype) for u in recent_dirs[-int(memory_cap):]], dim=0)
        b = B.mean(dim=0)
        bn = b.norm(p=2)
        if not torch.isfinite(bn) or bn <= 1e-12:
            return g
        b_unit = b / bn
        rp = float(penalty)
        if rp < 0.0:
            rp = 0.0
        if rp > 1.0:
            rp = 1.0
        q_min = 1e-12
        log_q = (1.0 - rp) * torch.log(torch.tensor(0.5, device=g.device, dtype=torch.float32))
        log_q = log_q + rp * torch.log(torch.tensor(q_min, device=g.device, dtype=torch.float32))
        q = torch.exp(log_q)
        p = torch.clamp(1.0 - q, 1e-12, 1.0 - 1e-12)
        beta = torch.sqrt(torch.tensor(2.0, device=g.device, dtype=g.dtype)) * torch.special.erfinv(2.0 * p - 1.0)
        beta = torch.clamp(beta, 0.0, 7.5)
        return g - (beta * b_unit)

    def _sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            g = torch.randn_like(final_hidden)
        else:
            g = torch.randn_like(final_hidden, generator=rng)
        if (
            self.repetition_penalty is not None
            and self.repetition_memory > 0
            and len(recent_dirs) > 0
        ):
            g = self._apply_repetition_penalty(g, float(self.repetition_penalty), int(self.repetition_memory), recent_dirs)
        return g / (g.norm(p=2) + 1e-12)
    
    def _update_base_vector(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
    ) -> torch.Tensor:
        return final_hidden


class IsotropicDirectionSampler(PenaltySampler):
    def __init__(self, repetition_penalty: float = 0.0, repetition_memory: int = 0, cfg: Optional[Dict] = None) -> None:
        # Intentionally ignore repetition controls; behave as a pure isotropic sampler
        super().__init__(repetition_penalty=0.0, repetition_memory=0, cfg=cfg)

    def _sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        g = torch.randn_like(final_hidden) if rng is None else torch.randn_like(final_hidden, generator=rng)
        return g / (g.norm(p=2) + 1e-12)

    def _update_base_vector(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
    ) -> torch.Tensor:
        return final_hidden

    def sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # Always return a unit vector; ignore to_noise_vector flag
        return self._sample_unit_direction(final_hidden, recent_dirs, rng=rng)

class VectorAnglePenalty(PenaltySampler):
    def __init__(self, repetition_penalty: float = 0.0, repetition_memory: int = 0, cfg: Optional[Dict] = None) -> None:
        super().__init__(repetition_penalty=repetition_penalty, repetition_memory=repetition_memory, cfg=cfg)

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2) + 1e-12)

    @staticmethod
    def _project(v: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
        # Project v onto onto (not necessarily unit); safe for tiny norms
        denom = onto.norm(p=2) ** 2 + 1e-12
        return (v.dot(onto) / denom) * onto

    @staticmethod
    def _compute_aggregate_axis(recent_dirs: List[torch.Tensor], device: torch.device, dtype: torch.dtype, memory_cap: int) -> Optional[torch.Tensor]:
        if memory_cap <= 0 or len(recent_dirs) == 0:
            return None
        B = torch.stack([u.to(device=device, dtype=dtype) for u in recent_dirs[-int(memory_cap):]], dim=0)
        mean_dir = B.mean(dim=0)
        if not torch.isfinite(mean_dir).all():
            return None
        if mean_dir.norm(p=2) <= 1e-12:
            return None
        return mean_dir

    def _rotation_amount(self, base_angle: torch.Tensor) -> torch.Tensor:
        # Nonlinear schedule with conservative per-step rotation:
        # - Rotation shrinks rapidly as the current angle grows (Gaussian decay).
        # - Rotation grows with repetition_penalty via arctan (slowly saturating),
        #   so only extremely large penalties approach large angles asymptotically.
        # - Hard-cap per step to ~5° to avoid instability.

        device = base_angle.device
        dtype = base_angle.dtype

        # rp_scale in [0, 1): arctan mapping (k controls growth speed)
        rp = 0.0 if self.repetition_penalty is None else float(self.repetition_penalty)
        rp = max(0.0, rp)
        k = torch.tensor(1.0, device=device, dtype=dtype)
        rp_t = torch.tensor(rp, device=device, dtype=dtype)
        rp_scale = torch.atan(k * rp_t) / (0.5 * torch.pi)

        # angle_weight decays with angle; beta ≈ 20° controls decay width
        beta = torch.tensor(torch.pi / 9.0, device=device, dtype=dtype)  # ~20°
        angle_weight = torch.exp(- (base_angle / beta) ** 2)

        # Uncapped desired rotation away from the axis
        raw_delta = (torch.tensor(torch.pi, device=device, dtype=dtype) - base_angle) * rp_scale * angle_weight

        # Per-step hard cap ≈ 5°
        delta_cap = torch.tensor(torch.pi / 36.0, device=device, dtype=dtype)
        delta = torch.clamp(raw_delta, min=0.0, max=delta_cap)
        return delta

    def _rotate_away_from_axis(self, g: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        # Decompose g into parallel and perpendicular to axis
        axis_unit = self._unit(axis)
        g_par = self._project(g, axis_unit)
        g_perp = g - g_par
        perp_norm = g_perp.norm(p=2)
        if perp_norm <= 1e-12:
            # g is almost collinear with axis; push into any orthogonal direction
            # Find an arbitrary orthogonal vector via Householder-like trick
            # Choose the smallest magnitude component to tweak for stability
            idx = torch.argmin(torch.abs(axis_unit))
            e = torch.zeros_like(axis_unit)
            e[idx] = 1.0
            g_perp = e - (e.dot(axis_unit)) * axis_unit
            g_par = g - g_perp
            perp_norm = g_perp.norm(p=2)
        u_perp = g_perp / (perp_norm + 1e-12)

        # Base angle between g and axis
        cos_theta = torch.clamp(self._unit(g).dot(axis_unit), -1.0, 1.0)
        base_angle = torch.arccos(cos_theta)
        # Compute rotation amount
        delta = self._rotation_amount(base_angle)

        # Rotate within the plane spanned by (axis_unit, u_perp):
        # new_g = cos(delta)*(g_par + g_perp) + sin(delta)*(+u_perp * ||g_perp||)
        # But to turn away from axis, we increase the perpendicular component magnitude
        new_par = torch.cos(delta) * g_par
        new_perp = g_perp + torch.sin(delta) * perp_norm * u_perp
        new_g = new_par + new_perp
        return new_g

    def _update_base_vector(
            self,
            final_hidden: torch.Tensor,
            recent_dirs: List[torch.Tensor],
        ) -> torch.Tensor:
        # Rotate final_hidden away from aggregate recent axis; preserve its norm
        if self.repetition_memory <= 0 or len(recent_dirs) == 0 or self.repetition_penalty <= 0.0:
            return final_hidden
        axis = self._compute_aggregate_axis(recent_dirs, final_hidden.device, final_hidden.dtype, self.repetition_memory)
        if axis is None:
            return final_hidden
        base_norm = final_hidden.norm(p=2)
        if base_norm <= 1e-12:
            return final_hidden
        u = self._unit(final_hidden)
        u_rot = self._rotate_away_from_axis(u, axis)
        return u_rot * base_norm

    def _sample_unit_direction(
        self,
        final_hidden: torch.Tensor,
        recent_dirs: List[torch.Tensor],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # Start with isotropic Gaussian proposal
        g = torch.randn_like(final_hidden) if rng is None else torch.randn_like(final_hidden, generator=rng)

        # For VectorAnglePenalty, rotation is applied to base vector, not to g

        return self._unit(g)
        

REGISTRY = {
    "mean_shift_penalty": DirectionWithMeanShiftPenalty,
    "vector_angle_penalty": VectorAnglePenalty,
    "isotropic": IsotropicDirectionSampler,
}


def build_direction_sampler(name: str, **kwargs) -> PenaltySampler:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown DirectionSampler: {name}")
    return cls(**kwargs)


