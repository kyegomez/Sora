import torch 
from torch import nn, Tensor
from einops import rearrange


def patchify_videos(
    x: Tensor,
    patch_size: tuple = (16, 16, 16),
):
    """
    Rearranges a video tensor into patches of specified size.

    Args:
        x (Tensor): The input video tensor of shape (batch_size, channels, depth, height, width).
        patch_size (tuple, optional): The size of each patch in the format (depth, height, width). Defaults to (16, 16, 16).

    Returns:
        Tensor: The rearranged video tensor of shape (batch_size * num_patches, channels, patch_depth, patch_height, patch_width).
    """
    d, h, w, = patch_size
    
    # Rearrange the video tensor into patches
    patched_videos = rearrange(
        x,
        "b c (d pd) (h ph) (w pw) -> (b d h w) c pd ph pw",
        pd=d, ph=h, pw=w
    )
    
    return patched_videos


class Sora(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x
