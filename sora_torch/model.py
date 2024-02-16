import torch
from einops import rearrange
from torch import Tensor, nn
from zeta.nn.attention import SpatialLinearAttention
from zeta.nn import FeedForward, video_to_text


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
    (
        d,
        h,
        w,
    ) = patch_size

    # Rearrange the video tensor into patches
    patched_videos = rearrange(
        x,
        "b c (d pd) (h ph) (w pw) -> (b d h w) c pd ph pw",
        pd=d,
        ph=h,
        pw=w,
    )

    return patched_videos


class AttentionBasedInflationBlock(nn.Module):
    """
    Attention-based inflation block module.

    Args:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout rate. Defaults to 0.1.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        attn (SpatialLinearAttention): The spatial linear ablttention module.
        proj (nn.Linear): The linear projection layer.
        norm (nn.LayerNorm): The layer normalization module.

    Example:
        >>> import torch
        >>> from lumiere.model import AttentionBasedInflationBlock
        >>> x = torch.randn(1, 4, 224, 224, 512)
        >>> model = AttentionBasedInflationBlock(dim=512, heads=4, dropout=0.1)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 4, 224, 224, 512])

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout

        # Spatial linear attention for videos of size:
        # batch_size, channels, frames, height, width.
        self.attn = SpatialLinearAttention(
            dim, heads, dim_head=dim // heads, *args, **kwargs
        )

        # Linear projection layer
        self.proj = nn.Linear(dim, dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the AttentionBasedInflationBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        skip = x
        b, t, h, w, d = x.shape

        # Reshape to match the spatial linear attention module
        x = rearrange(x, "b t h w d -> b d t h w")

        # Apply spatial linear attention
        x = self.attn(x)

        # Reshape back to the original shape
        x = rearrange(x, "b d t h w -> b t h w d")

        # Linear projection
        x = nn.Linear(d, d)(x)

        return x + skip


class VideoCompressionViT(nn.Module):
    """
    VideoCompressionViT model.

    Args:
        dim (int): Dimension of the input tensor.
        num_patches (int): Number of patches to divide the input tensor into.
        depth (int): Depth of the model.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the feedforward network.
        channels (int): Number of input channels.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        dim: int,
        num_patches: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dropout = dropout

        # Model
        self.model = AttentionBasedInflationBlock(
            dim, heads, dropout, *args, **kwargs
        )

        # FeedForward
        self.ffn = FeedForward(dim, dim, mlp_dim, *args, **kwargs)

    def forward(self, x: Tensor):
        """VideoCompressionViT forward pass.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            Tensor: The output tensor.
        """
        b, c, t, h, w = x.shape
        # x = patchify_videos(
        #     x, (self.num_patches, self.num_patches, self.num_patches)
        # )
        # print(x.shape)
        # Attention-based inflation block
        x = self.model(x)

        # FFN
        x = video_to_text(x, self.num_patches, self.dim, True)

        # FeedForward
        x = self.ffn(x)

        # Rearrange back to the original shape
        return x


x = torch.randn(1, 4, 224, 224, 512)
model = VideoCompressionViT(
    dim=512,
    num_patches=4,
    depth=4,
    heads=4,
    mlp_dim=512,
    channels=4,
    dropout=0.1,
)
out = model(x)
print(out.shape)


class Sora(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x
