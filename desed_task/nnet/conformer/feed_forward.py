import torch
from .activation import Swish


class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        Pdrop: dropout probability
        act: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(self, dim_model, dim_ffn, Pdrop, act, inner_dropout):
        super(FeedForwardModule, self).__init__()

        # Assert
        assert act in ["relu", "swish"]

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            nn.Linear(dim_model, dim_ffn),
            Swish() if act=="swish" else nn.ReLU(),
            nn.Dropout(p=Pdrop) if inner_dropout else nn.Identity(),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(p=Pdrop)
        )

    def forward(self, x):
        return self.layers(x)