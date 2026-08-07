"""DDPM (Ho, Jain & Abbeel, 2020) -- first and only Diffusion-family
architecture. Real reverse-diffusion sampling (not training): a small
noise-prediction CNN (eps_theta) plus the DDPM ancestral sampling formula,
run for T=20 steps rather than the paper's 1000 -- fewer steps is a real,
common practical tradeoff (production DDIM/DDPM samplers routinely use
10-50 steps), and this project's models are deliberately small/fast
everywhere (BERT: 2 layers, GCN: hidden_dim=16) since the point is
realistic traffic, not sample quality.

Unlike every other architecture in this project, "inference" here is a
T-step loop, not one forward pass -- but that loop lives entirely inside
_DDPMSampler.forward(), so it's invisible to frameworks/pytorch_adapter.py
and the rest of the pipeline: one predict() call in, one generated image
out, over the same wire protocol as everything else. That's what makes
this fit the existing single-request/response inference paradigm without
any change to core/roles code -- diffusion model SAMPLING (as opposed to
training) genuinely is request/response shaped: give it noise, get back
an image.

Default build is this hand-rolled sampler; when `diffusion_framework:
Diffusers` is set in the config, build() dispatches to
diffusion_frameworks/diffusers_adapter.py's real UNet2DModel/DDPMScheduler-
based sampler instead -- see that module's docstring for the real bug
(norm_num_groups) found and fixed while verifying it.
"""
import torch

from core.registry import ARCHITECTURES
from families.diffusion import linear_beta_schedule

_NUM_TIMESTEPS = 20


class _NoisePredictor(torch.nn.Module):
    """Small conv net predicting the noise added at timestep t, with a
    learned per-timestep embedding broadcast-added into the feature maps."""

    def __init__(self, channels=3, hidden_channels=32, num_timesteps=_NUM_TIMESTEPS):
        super().__init__()
        self.time_embedding = torch.nn.Embedding(num_timesteps, hidden_channels)
        self.conv_in = torch.nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.conv_mid = torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv_out = torch.nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.activation = torch.nn.SiLU()

    def forward(self, x, t):
        time_emb = self.time_embedding(t)[:, :, None, None]  # (B, hidden, 1, 1)
        h = self.activation(self.conv_in(x) + time_emb)
        h = self.activation(self.conv_mid(h) + time_emb)
        return self.conv_out(h)


class _DDPMSampler(torch.nn.Module):
    def __init__(self, num_timesteps=_NUM_TIMESTEPS):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.noise_predictor = _NoisePredictor(num_timesteps=num_timesteps)
        betas, alphas, alphas_cumprod = linear_beta_schedule(num_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x_t):
        """x_t (B, 3, 32, 32) is treated as the initial pure-noise sample
        x_T; runs the full ancestral-sampling reverse process and returns
        x_0, the generated image."""
        x = x_t
        batch_size = x.shape[0]
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=x.device)
            predicted_noise = self.noise_predictor(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            )
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x


def build(framework_adapter, config):
    diffusion_framework = getattr(config, "diffusion_framework", None)
    if diffusion_framework:
        from core.registry import DIFFUSION_FRAMEWORKS

        adapter = DIFFUSION_FRAMEWORKS.get(diffusion_framework).build()
        if not hasattr(adapter, "build_ddpm"):
            raise RuntimeError(
                f"diffusion_framework '{diffusion_framework}' has no DDPM "
                f"implementation to dispatch to (see architectures/ddpm.py)."
            )
        return adapter.build_ddpm()
    return _DDPMSampler()


ARCHITECTURES.register(
    "DDPM",
    implemented=True,
    family="Diffusion",
    framework="PyTorch",
    application="Image Generation",
    input_shape=(3, 32, 32),
)(build)
