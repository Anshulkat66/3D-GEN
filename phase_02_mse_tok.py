import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
    
class MultiScaleEEGEncoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()

        # 🔥 Multi-scale branches
        self.small_scale = ConvBlock(in_channels, hidden_dim, kernel_size=3)
        self.medium_scale = ConvBlock(in_channels, hidden_dim, kernel_size=7)
        self.large_scale = ConvBlock(in_channels, hidden_dim, kernel_size=15)

        # Combine scales
        self.fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]

        s = self.small_scale(x)
        m = self.medium_scale(x)
        l = self.large_scale(x)

        # Concatenate
        combined = torch.cat([s, m, l], dim=1)

        # Fuse
        out = self.fusion(combined)

        return out
    

class Tokenizer(nn.Module):
    def __init__(self, in_dim=128, embed_dim=256, stride=4):
        super().__init__()

        self.token_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=embed_dim,
            kernel_size=stride,
            stride=stride
        )

    def forward(self, x):
        # x: [B, C, T]

        tokens = self.token_conv(x)

        # convert to sequence
        tokens = tokens.permute(0, 2, 1)  # [B, T_tokens, embed_dim]

        return tokens


class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_scale = MultiScaleEEGEncoder(
            in_channels=64,
            hidden_dim=128
        )

        self.tokenizer = Tokenizer(
            in_dim=128,
            embed_dim=256,
            stride=4
        )

    def forward(self, x):
        # x: [B, T, C]

        # 🔥 convert to conv format
        x = x.permute(0, 2, 1)  # [B, C, T]

        features = self.multi_scale(x)

        tokens = self.tokenizer(features)

        return tokens
    
if __name__ == "__main__":
    model = EEGEncoder()

    dummy = torch.randn(2, 1024, 64)

    out = model(dummy)

    print("Output shape:", out.shape)