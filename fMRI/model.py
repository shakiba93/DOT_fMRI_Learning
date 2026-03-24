import torch
from torch import nn
import torch.nn.functional as F


class Bold_Hybrid(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        num_layers=4,
        num_classes=15,
        time_steps=87,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Separate projections for HbO and HbR
        self.project_hbo = nn.Linear(371, embedding_dim)
        self.project_hbr = nn.Linear(371, embedding_dim)

        self.positional_encoding = nn.Parameter(
            torch.randn(1, time_steps, embedding_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=0.25,
            batch_first=True,
            dim_feedforward=256,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (B, 2, 371, 87)

        hbo = x[:, 0]   # (B, 371, 87)
        hbr = x[:, 1]   # (B, 371, 87)

        # move time to middle
        hbo = hbo.permute(0, 2, 1)  # (B, 87, 371)
        hbr = hbr.permute(0, 2, 1)  # (B, 87, 371)

        # separate embeddings
        hbo_emb = self.project_hbo(hbo)  # (B, 87, D)
        hbr_emb = self.project_hbr(hbr)  # (B, 87, D)

        # 🔹 CRUCIAL HYBRID STEP
        x = (hbo_emb + hbr_emb) / 2      # shared representation

        x = x + self.positional_encoding

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.dropout(x)
        return self.classification(x)
