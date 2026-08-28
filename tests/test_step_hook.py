"""Behavioral tests for ARDY's cooperative denoising-step hook.

The fixtures below keep the source generation loop real while replacing only
the heavyweight model components.  This lets the tests prove callback timing,
ordering, output parity, and exception propagation without downloading a
checkpoint or changing the denoising algorithm.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from ardy.model.ardy_model import Ardy


class _IdentityStats:
    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _TinyDiffusion:
    def space_timesteps(self, num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(int(num_steps), dtype=torch.long)
        return steps, steps

    def calc_diffusion_vars(self, _use_timesteps: torch.Tensor) -> None:
        return None


class _TinyHybrid:
    def __init__(self) -> None:
        self.explicit_calls = 0

    def convert_frame_mask_to_token_mask(self, *masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return masks[:3]

    def get_root_and_latent_body_motion_from_hybrid(
        self,
        hybrid_motion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = hybrid_motion.shape
        root_motion = torch.zeros(
            batch_size,
            token_count * 4,
            3,
            dtype=hybrid_motion.dtype,
            device=hybrid_motion.device,
        )
        latent_body_motion = hybrid_motion[:, :, 1:]
        return root_motion, latent_body_motion

    def get_hybrid_motion_from_root_and_latent_body_motion(
        self,
        root_motion: torch.Tensor,
        latent_body_motion: torch.Tensor,
    ) -> torch.Tensor:
        root_tokens = root_motion[:, ::4, :1]
        return torch.cat([root_tokens, latent_body_motion], dim=-1)

    def get_explicit_motion_from_hybrid(
        self,
        hybrid_motion: torch.Tensor,
        _motion_pad_mask: torch.Tensor,
        _motion_len: torch.Tensor,
        motion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del motion_mask
        self.explicit_calls += 1
        return hybrid_motion.repeat_interleave(4, dim=1)


class _TinyArdy(Ardy):
    """Minimal ARDY object that executes the production denoising loop."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.gen_horizon_len = 4
        self.num_frames_per_token = 4
        self.denoiser = SimpleNamespace(latent_embedding_dim=2, nframe_root_dim=1)
        self.motion_rep = SimpleNamespace(
            nfeats_dict={"root_pos": 3},
            global_root_stats=_IdentityStats(),
            slice_dict={"root_pos": slice(0, 3)},
        )
        self.autoencoder = SimpleNamespace(encode_with_quantization=False)
        self.diffusion = _TinyDiffusion()  # type: ignore[assignment]
        self.hybrid = _TinyHybrid()
        self.denoising_timesteps: list[int] = []
        self.denoising_grad_enabled: list[bool] = []

    def denoising_step(self, x: torch.Tensor, *args: object, **_kwargs: object) -> torch.Tensor:
        timestep = args[11]
        assert isinstance(timestep, torch.Tensor)
        self.denoising_timesteps.append(int(timestep.item()))
        self.denoising_grad_enabled.append(torch.is_grad_enabled())
        return x + timestep.to(dtype=x.dtype).reshape(1, 1, 1)


def _run_window(step_hook: Callable[[int], None] | None) -> tuple[_TinyArdy, torch.Tensor]:
    torch.manual_seed(20260827)
    model = _TinyArdy()
    output = model._generate_window(
        history_sequence=None,
        global_transl=torch.zeros((1, 3)),
        history_start_frame=0,
        history_end_frame=0,
        total_frames=4,
        text_feat=torch.zeros((1, 1, 2)),
        text_pad_mask=torch.zeros((1, 1), dtype=torch.bool),
        first_heading_angle=None,
        motion_mask=None,
        observed_motion=None,
        num_denoising_steps=torch.tensor([6]),
        cfg_weight=2.0,
        indices=list(range(6))[::-1],
        progress_bar=lambda values: values,
        step_hook=step_hook,
    )
    return model, output


def _run_autoregressive_step(
    model: _TinyArdy,
    step_hook: Callable[[int], None] | None,
) -> torch.Tensor:
    return model.autoregressive_step(
        num_frames=4,
        num_denoising_steps=6,
        motion_mask=None,
        observed_motion=None,
        text_feat=torch.zeros((1, 1, 2)),
        text_pad_mask=torch.zeros((1, 1), dtype=torch.bool),
        step_hook=step_hook,
    )


def test_absent_and_noop_hooks_preserve_identical_window_tensor_bytes() -> None:
    plain_model, plain = _run_window(None)
    noop_model, noop = _run_window(lambda _step: None)
    observed_steps: list[int] = []
    hooked_model, hooked = _run_window(lambda step: observed_steps.append(step))

    assert torch.equal(plain, noop)
    assert torch.equal(plain, hooked)
    assert plain.contiguous().numpy().tobytes() == noop.contiguous().numpy().tobytes()
    assert plain.contiguous().numpy().tobytes() == hooked.contiguous().numpy().tobytes()
    assert plain_model.denoising_timesteps == noop_model.denoising_timesteps == hooked_model.denoising_timesteps
    assert observed_steps == list(range(1, 7))
    assert plain_model.denoising_timesteps == [5, 4, 3, 2, 1, 0]


def test_autoregressive_step_calls_hook_after_each_reversed_denoising_timestep() -> None:
    model = _TinyArdy()
    observed_steps: list[int] = []
    hook_grad_enabled: list[bool] = []

    def observe_step(step: int) -> None:
        observed_steps.append(step)
        hook_grad_enabled.append(torch.is_grad_enabled())

    output = _run_autoregressive_step(model, observe_step)

    assert output.shape == (1, 4, 3)
    assert observed_steps == list(range(1, 7))
    assert model.denoising_timesteps == [5, 4, 3, 2, 1, 0]
    assert model.denoising_grad_enabled == [False] * 6
    assert hook_grad_enabled == [True] * 6
    assert model.hybrid.explicit_calls == 1


def test_autoregressive_step_noop_hook_preserves_deterministic_output_bytes() -> None:
    torch.manual_seed(20260827)
    plain = _run_autoregressive_step(_TinyArdy(), None)
    torch.manual_seed(20260827)
    noop = _run_autoregressive_step(_TinyArdy(), lambda _step: None)

    assert torch.equal(plain, noop)
    assert plain.contiguous().numpy().tobytes() == noop.contiguous().numpy().tobytes()


def test_callback_exception_aborts_autoregressive_step_before_partial_output() -> None:
    class StepAbort(RuntimeError):
        pass

    model = _TinyArdy()
    observed_steps: list[int] = []
    published: list[torch.Tensor] = []

    def abort_at_third_step(step: int) -> None:
        observed_steps.append(step)
        if step == 3:
            raise StepAbort("deadline at denoising boundary 3")

    with pytest.raises(StepAbort, match="deadline at denoising boundary 3"):
        published.append(_run_autoregressive_step(model, abort_at_third_step))

    assert observed_steps == [1, 2, 3]
    assert model.denoising_timesteps == [5, 4, 3]
    assert model.hybrid.explicit_calls == 0
    assert published == []
