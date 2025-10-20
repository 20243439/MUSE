import torch
import torch.nn as nn

from src.encoder.ffn_encoder import FFNEncoder
from src.encoder.image_encoder import ImageEncoder
from gnn import MML


class PADUFESBackbone(nn.Module):
    def __init__(
            self,
            embedding_size,
            dropout,
            ffn_layers,
            gnn_layers,
            gnn_norm=None,
            device="cpu",
            image_freeze: bool = False,
            num_classes: int = 6,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        # Image encoder (x1)
        self.x1_encoder = ImageEncoder(output_dim=embedding_size, pretrained=True, freeze=image_freeze)
        self.x1_mapper = nn.Linear(embedding_size, embedding_size)

        # Tabular encoder (x2). Input dim will be set dynamically on first forward
        self.x2_encoder = None
        self.x2_mapper = nn.Linear(embedding_size, embedding_size)

        # x3 unused but placeholder kept for API compatibility
        self.x3_encoder = None
        self.x3_mapper = nn.Identity()

        self.mml = MML(
            num_modalities=3,
            hidden_channels=embedding_size,
            num_layers=gnn_layers,
            dropout=dropout,
            normalize_embs=gnn_norm,
            num_classes=num_classes,
        )

    def _ensure_tabular_encoder(self, x2):
        if self.x2_encoder is None:
            input_dim = x2.size(-1)
            self.x2_encoder = FFNEncoder(
                input_dim=input_dim,
                hidden_dim=self.embedding_size,
                output_dim=self.embedding_size,
                num_layers=self.ffn_layers,
                dropout_prob=self.dropout,
                device=self.device,
            ).to(self.device)

    def forward(self, x1, x1_flag, x2, x2_flag, x3, x3_flag, label, label_flag, **kwargs):
        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)
        x3_flag = x3_flag.to(self.device)
        label = label.to(self.device)
        label_flag = label_flag.to(self.device)

        # Image
        x1 = x1.to(self.device)
        x1_embedding = self.x1_encoder(x1)
        x1_embedding = self.x1_mapper(x1_embedding)
        x1_embedding[x1_flag == 0] = 0
        x1_embedding = self.dropout_layer(x1_embedding)

        # Tabular
        x2 = x2.to(self.device)
        self._ensure_tabular_encoder(x2)
        x2_embedding = self.x2_encoder(x2)
        x2_embedding = self.x2_mapper(x2_embedding)
        x2_embedding[x2_flag == 0] = 0
        x2_embedding = self.dropout_layer(x2_embedding)

        # x3 placeholder
        bsz = x1.size(0)
        x3_embedding = torch.zeros(bsz, self.embedding_size, device=self.device)

        loss = self.mml(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
            label, label_flag,
        )
        return loss

    def inference(self, x1, x1_flag, x2, x2_flag, x3, x3_flag, **kwargs):
        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)
        x3_flag = x3_flag.to(self.device)

        # Image
        x1 = x1.to(self.device)
        x1_embedding = self.x1_encoder(x1)
        x1_embedding = self.x1_mapper(x1_embedding)
        x1_embedding[x1_flag == 0] = 0

        # Tabular
        x2 = x2.to(self.device)
        self._ensure_tabular_encoder(x2)
        x2_embedding = self.x2_encoder(x2)
        x2_embedding = self.x2_mapper(x2_embedding)
        x2_embedding[x2_flag == 0] = 0

        # x3 placeholder
        bsz = x1.size(0)
        x3_embedding = torch.zeros(bsz, self.embedding_size, device=self.device)

        y_scores, logits = self.mml.inference(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
        )
        return y_scores, logits
