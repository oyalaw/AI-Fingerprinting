"""Diffusers -- real, verified implementation of a DDPM sampler equivalent
to architectures/ddpm.py's hand-rolled one, built with HuggingFace's real
`UNet2DModel` + `DDPMScheduler` instead of a hand-written noise-prediction
CNN and manually-coded reverse-sampling formula.

**Honest disclosure on wiring**: exactly the same situation as
graph_frameworks/pytorch_geometric_adapter.py -- this is NOT currently
reachable by setting `diffusion_framework: Diffusers` in a config.
`architecture_entry.build()` only ever receives the selected framework
adapter, never the full `ExperimentConfig`, so there's no branch point for
architectures/ddpm.py to consult which diffusion_framework was selected.
Making that real means threading `config` through `FrameworkAdapter.
load_model()` and every architecture's `build()` across every implemented
framework adapter and architecture -- a real, separate refactor, not
folded in unprompted alongside one adapter. `core/config.py` already
validates `diffusion_framework` against this registry; this file makes
that validated value point at something real for the first time.

**A real bug found and fixed while verifying this, not just code that
happens to run**: `UNet2DModel`'s default `norm_num_groups=32` doesn't
divide evenly into this project's deliberately small `block_out_channels`
(16, 32) -- construction raises `ValueError: num_channels (16) must be
divisible by num_groups (32)`. Fixed with an explicit `norm_num_groups=8`
(16/8=2, 32/8=4, both clean).

Confirmed directly: a full T=20-step reverse-diffusion loop (same T as
architectures/ddpm.py, for the same "small and fast, not paper-quality"
reason) via `DDPMScheduler.set_timesteps`/`.step()` produces a correctly-
shaped (1, 3, 32, 32) output with no NaN/Inf, from pure Gaussian noise.

diffusers is a pure-Python wheel -- confirmed via `diffusers-*-py3-none-
any.whl`, no platform-specific build. Its bundled `diffusers-cli.exe` is
deliberately never invoked here; only the library's own Python classes
(`UNet2DModel`, `DDPMScheduler`) are used.
"""
import torch

from core.registry import DIFFUSION_FRAMEWORKS

_NUM_TIMESTEPS = 20


class _DiffusersDDPMSampler(torch.nn.Module):
    def __init__(self, num_timesteps=_NUM_TIMESTEPS):
        super().__init__()
        from diffusers import DDPMScheduler, UNet2DModel

        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(16, 32),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
        self.num_timesteps = num_timesteps

    def forward(self, x_t):
        """x_t (B, 3, 32, 32) is treated as the initial pure-noise sample
        x_T; runs the full DDPMScheduler reverse process and returns x_0."""
        x = x_t
        self.scheduler.set_timesteps(self.num_timesteps)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(x, t).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample
        return x


class DiffusersAdapter:
    """Not a FrameworkAdapter (frameworks/base.py) -- diffusion_frameworks/
    has no base.py of its own yet, since nothing in roles/client.py or
    roles/server.py consults this registry (see module docstring)."""

    def build_ddpm(self, num_timesteps=_NUM_TIMESTEPS):
        return _DiffusersDDPMSampler(num_timesteps)


@DIFFUSION_FRAMEWORKS.register("Diffusers", implemented=True, organization="HuggingFace")
def build_diffusers_adapter(**kwargs):
    return DiffusersAdapter()
