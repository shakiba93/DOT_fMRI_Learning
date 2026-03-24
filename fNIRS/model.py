import torch
import glob
import pickle
import numpy as np
from torch import nn

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pathlib import PureWindowsPath
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np

class CNN2D(nn.Module):
    def __init__(
        self,
        in_channels=104,
        num_classes=2,
        n_modalities=2,
        emb_dim=64
    ):
        super().__init__()

        self.modality_emb = nn.Embedding(n_modalities, emb_dim)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.classifier = None
        self.num_classes = num_classes

    def forward(self, x, modality):

        # First conv
        x = self.feature_extractor[0](x)

        # modality embedding (currently unused but kept)
        emb = self.modality_emb(modality)
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        # x = x + emb

        # rest of convs
        for layer in self.feature_extractor[1:]:
            x = layer(x)

        # dynamic classifier
        if self.classifier is None:
            flattened_size = x.view(x.size(0), -1).size(1)

            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, self.num_classes)
            )
            self.classifier.to(x.device)

        return self.classifier(x)

#### keeep
class CNN2DImageWustl(nn.Module):
    def __init__(self, n_modalities=2, emb_dim=64):
        super().__init__()

        # modality embedding (one vector per modality)
        self.modality_emb = nn.Embedding(n_modalities, emb_dim)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(371, 64, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.classifier = None

    def forward(self, x, modality):
        """
        x: (B, 104, 1, T)
        modality: (B,) long tensor, values in {0,1}
        """

        # ---- First conv ----
        x = self.feature_extractor[0](x)   # Conv2d -> (B, 64, 1, T')

        # ---- Inject modality embedding ----
        emb = self.modality_emb(modality)  # (B, 64)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        # x = x + emb

        # ---- Rest of feature extractor ----
        for layer in self.feature_extractor[1:]:
            x = layer(x)

        # ---- Dynamic classifier ----
        if self.classifier is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x


### keeeep
class CNN2DImageWustlNew(nn.Module):
    def __init__(self, n_modalities=2, emb_dim=64):
        super().__init__()

        # modality embedding (one vector per modality)
        self.modality_emb = nn.Embedding(n_modalities, emb_dim)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(371, 64, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.classifier = None

    def forward(self, x, modality):
        """
        x: (B, 104, 1, T)
        modality: (B,) long tensor, values in {0,1}
        """

        # ---- First conv ----
        x = self.feature_extractor[0](x)   # Conv2d -> (B, 64, 1, T')

        # ---- Inject modality embedding ----
        emb = self.modality_emb(modality)  # (B, 64)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        # x = x + emb

        # ---- Rest of feature extractor ----
        for layer in self.feature_extractor[1:]:
            x = layer(x)

        # ---- Dynamic classifier ----
        if self.classifier is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 5)
            )
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x

#### keeep
class CNN2DImage(nn.Module):
    def __init__(self, n_modalities=2, emb_dim=64):
        super().__init__()

        # modality embedding (one vector per modality)
        self.modality_emb = nn.Embedding(n_modalities, emb_dim)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(104, 64, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        self.classifier = None

    def forward(self, x, modality):
        """
        x: (B, 104, 1, T)
        modality: (B,) long tensor, values in {0,1}
        """

        # ---- First conv ----
        x = self.feature_extractor[0](x)   # Conv2d -> (B, 64, 1, T')

        # ---- Inject modality embedding ----
        emb = self.modality_emb(modality)  # (B, 64)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        # x = x + emb

        # ---- Rest of feature extractor ----
        for layer in self.feature_extractor[1:]:
            x = layer(x)

        # ---- Dynamic classifier ----
        if self.classifier is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x

#### keeep
class Bold_Hybrid(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        num_layers=4,
        num_classes=15,
        time_steps=87,
        dropout=0.5,
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
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        return self.classification(x)

