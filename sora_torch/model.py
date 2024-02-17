import torch
from einops import rearrange
from torch import Tensor, nn
from zeta.nn.attention import (
    SpatialLinearAttention,
    MultiQueryAttention,
)
from zeta.nn import FeedForward
from einops.layers.torch import Rearrange


class PatchEmbeddingLatentSpace(nn.Module):
    """
    PatchEmbeddingLatentSpace module for converting image patches into a latent space representation.

    Args:
        in_channels (int): Number of input channels.
        patch_size (int): Size of each patch.
        dim (int): Dimension of the latent space representation.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        in_channels (int): Number of input channels.
        patch_size (int): Size of each patch.
        dim (int): Dimension of the latent space representation.
        proj (nn.Sequential): Sequential module for projecting patches into the latent space.

    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim

        self.proj = nn.Sequential(
            Rearrange(
                "b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)",
                pd=patch_size,
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(patch_size**3 * in_channels, dim),
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the PatchEmbeddingLatentSpace module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Latent space representation of the input tensor.

        """
        x = self.proj(x)

        return x


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

        # Attn
        self.attn = MultiQueryAttention(
            dim,
            heads,
            # qk_ln=True,
        )

        # PatchEmbeddingLatentSpace
        self.patch_embed = PatchEmbeddingLatentSpace(
            channels, num_patches, dim
        )

    def forward(self, x: Tensor):
        """VideoCompressionViT forward pass.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            Tensor: The output tensor.
        """
        b, c, t, h, w = x.shape

        # Embed patches
        # x = self.patch_embed(x)

        # print(x.shape)
        # Attention-based inflation block
        x = self.model(x)
        x, _ = self.attn(x, x, x)

        # FFN
        # x = video_to_text(x, self.num_patches, self.dim, True)

        # FeedForward
        x = self.ffn(x)

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
