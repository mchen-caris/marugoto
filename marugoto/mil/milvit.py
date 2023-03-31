#This model is based on the Vision Transformer model you provided but includes a simple attention mechanism to adapt it to a multi-instance learning scenario.
#In this adapted model, the Vision Transformer (ViT) is utilized as a feature extractor. The simple attention mechanism computes the importance of instances, and the final classifier outputs the class probabilities. This architecture should be suitable for multi-instance learning tasks in histopathology data.


import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from transformer import Transformer, ViT


class SimpleAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SimpleAttention, self).__init__()
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


class MILViT(nn.Module):
    def __init__(self, num_classes, input_dim=768, attention_dim=64):
        super(MILViT, self).__init__()

        # ViT model
        self.vit = ViT(num_classes=num_classes, input_dim=input_dim)

        # Remove the classifier head from the original ViT
        self.vit = nn.Sequential(*list(self.vit.children())[:-1])

        # Attention mechanism
        self.attention = SimpleAttention(input_dim, attention_dim)

        # Classifier head
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)

        attention_weights = self.attention(x)
        x = torch.sum(x * attention_weights, dim=1)

        x = self.classifier(x)
        return x
