import torch
import torch.nn as nn
import timm
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.input_dim = input_dim

        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        attention_weights = self.attention_layer(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MILModel(nn.Module):
    def __init__(self, n_classes, input_dim=1024, attention_dim=64, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(MILModel, self).__init__()

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout,
                                                 activation='relu', layer_norm_eps=1e-5)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=512, bidirectional=True, batch_first=True)
        self.lstm_norm = nn.LayerNorm(1024)

        # Attention mechanism
        self.attention = Attention(input_dim=1024, attention_dim=attention_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x, register_hook=False):
        batch_size, seq_length, _ = x.size()

        # Transformer Encoder
        x = x.view(batch_size, seq_length, -1)  # Flatten the feature matrices: (batch_size, seq_length, input_dim)
        x = x.transpose(0, 1)  # Transpose for Transformer: (seq_length, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Transpose back: (batch_size, seq_length, input_dim)

        # BiLSTM
        x, _ = self.bilstm(x)
        x = self.lstm_norm(x)

        attention_weights = self.attention(x)
        x = torch.sum(x * attention_weights, dim=1)

        x = self.classifier(x)
        return x