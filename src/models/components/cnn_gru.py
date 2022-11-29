import torch
import torch.nn as nn


class CnnGRUModel(nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        n_classes: int = 500,
        use_border: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_border = use_border

        self.video_encoder = encoder()

        # prepare for GRU input
        in_gru = self.video_encoder.outplanes
        if self.use_border:
            in_gru += 1

        self.gru = nn.GRU(
            in_gru, gru_hidden, gru_layers, batch_first=True, bidirectional=True, dropout=0.2
        )
        self.classifier = nn.Linear(gru_hidden * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, border=None):
        self.gru.flatten_parameters()

        f_v = self.video_encoder(x)
        f_v = self.dropout(f_v)
        f_v = f_v.float()

        if self.use_border:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], dim=-1))
        else:
            h, _ = self.gru(f_v)

        y_v = self.classifier(self.dropout(h)).mean(1)
        return y_v
