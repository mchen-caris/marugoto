#This alternative architecture utilizes a pre-trained CNN (e.g., ResNet) to extract features, followed by a Bi-directional LSTM (BiLSTM) for sequence modeling, and an attention mechanism
#This model first extracts features using a pre-trained CNN (ResNet50) and then uses a BiLSTM for sequence modeling. The attention mechanism computes the importance of the instances, and the final classifier outputs the class probabilities.

import torch
import torch.nn as nn
import torchvision.models as models


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


class MILModel(nn.Module):
    def __init__(self, n_classes, attention_dim=64):
        super(MILModel, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Linear(512, 2048)

        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(input_size=2048, hidden_size=512, bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = Attention(input_dim=1024, attention_dim=attention_dim)

        # Classifier head
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, x, register_hook=False):
        batch_size, seq_length, input_dim = x.size()

        # Feature extraction
        x = x.view(batch_size * seq_length, input_dim)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_length, -1)

        x, _ = self.bilstm(x)

        attention_weights = self.attention(x)
        x = torch.sum(x * attention_weights, dim=1)

        x = self.classifier(x)
        return x
