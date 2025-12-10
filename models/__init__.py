from .gan_mfs import GANMFSSynthesizer
from .wgan_gp import WGANSynthesizer
from .ctab_gan_plus import CTABGANSynthesizer as CTABGANPlusSynthesizer # когда подключишь

__all__ = [
    "GANMFSSynthesizer",
    "WGANSynthesizer",
    "CTABGANPlusSynthesizer",
]
