import torch
import torch.nn as nn

from src.encoder.ffn_encoder import FFNEncoder
from src.encoder.image_encoder import ImageEncoder
from src.encoder.text_encoder import TextEncoder
from gnn import MML


class YELPBackbone(nn.Module):
    def __init__(
        self,
        embedding_size,
        dropout,
        ffn_layers,
        gnn_layers,
        gnn_norm=None,
        device="cpu",
        image_freeze: bool = False,
        num_classes: int = 5,
        bert_type: str = "prajjwal1/bert-tiny",
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        # Image encoder (x1): ResNet50 pretrained
        self.x1_encoder = ImageEncoder(output_dim=embedding_size, pretrained=True, freeze=image_freeze)
        self.x1_mapper = nn.Linear(embedding_size, embedding_size)

        # Tabular encoder (x2): lazy init based on feature dim
        self.x2_encoder = None
        self.x2_mapper = nn.Linear(embedding_size, embedding_size)

        # Text encoder (x3): caption via transformers
        self.x3_encoder = TextEncoder(bert_type=bert_type, device=device)
        self.x3_mapper = nn.Linear(self.x3_encoder.model.config.hidden_size, embedding_size)

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
        x1_flag = x1_flag.to(self.device) if isinstance(x1_flag, torch.Tensor) else torch.tensor(x1_flag, device=self.device)
        x2_flag = x2_flag.to(self.device) if isinstance(x2_flag, torch.Tensor) else torch.tensor(x2_flag, device=self.device)
        x3_flag = x3_flag.to(self.device) if isinstance(x3_flag, torch.Tensor) else torch.tensor(x3_flag, device=self.device)
        label = label.to(self.device)
        label_flag = label_flag.to(self.device) if isinstance(label_flag, torch.Tensor) else torch.tensor(label_flag, device=self.device)

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

        # Text
        if isinstance(x3, list):
            texts = x3
        else:
            texts = [x3]
        text_emb = self.x3_encoder(texts)  # [B, H]
        x3_embedding = self.x3_mapper(text_emb)
        # x3_flag may be bool per-sample; expand to tensor mask
        if not isinstance(x3_flag, torch.Tensor):
            x3_flag = torch.tensor([bool(t) and len(t) > 0 for t in texts], device=self.device)
        x3_embedding[x3_flag == 0] = 0
        x3_embedding = self.dropout_layer(x3_embedding)

        loss = self.mml(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
            label, label_flag,
        )
        return loss

    def inference(self, x1, x1_flag, x2, x2_flag, x3, x3_flag, **kwargs):
        x1_flag = x1_flag.to(self.device) if isinstance(x1_flag, torch.Tensor) else torch.tensor(x1_flag, device=self.device)
        x2_flag = x2_flag.to(self.device) if isinstance(x2_flag, torch.Tensor) else torch.tensor(x2_flag, device=self.device)
        x3_flag = x3_flag.to(self.device) if isinstance(x3_flag, torch.Tensor) else torch.tensor(x3_flag, device=self.device)

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

        # Text
        texts = x3 if isinstance(x3, list) else [x3]
        text_emb = self.x3_encoder(texts)
        x3_embedding = self.x3_mapper(text_emb)
        if not isinstance(x3_flag, torch.Tensor):
            x3_flag = torch.tensor([bool(t) and len(t) > 0 for t in texts], device=self.device)
        x3_embedding[x3_flag == 0] = 0

        y_scores, logits = self.mml.inference(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
        )
        return y_scores, logits
