"""Diffusion family: shared linear beta noise schedule reused by every
diffusion architecture (DDPM, ...). Ho et al. 2020's schedule: beta_t
increases linearly from beta_1 to beta_T across T timesteps; alpha_t =
1 - beta_t; alpha_bar_t is the cumulative product of alphas up to t --
the quantity the reverse-sampling formula needs at inference time.
"""
import torch

from core.registry import FAMILIES


def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


FAMILIES.add("Diffusion", implemented=True, description="Diffusion Models")
