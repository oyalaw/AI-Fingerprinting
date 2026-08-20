"""GAN family: no shared preprocessing helper needed -- architectures/dcgan.py
reuses applications/image_generation.py's existing Gaussian-noise
preprocessing unchanged (same convention architectures/ddpm.py's Diffusion
family already established: give the model noise, get back an image)."""
from core.registry import FAMILIES

FAMILIES.add("GAN", implemented=True, description="Generative Adversarial Networks")
