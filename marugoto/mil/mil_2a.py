import torch
from typing import Optional

import torch
from torch import nn
__all__ = ["MILModel", "Attention"]

def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))

class MILModel(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        attention2: Optional[nn.Module] = None,  # New attention layer
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256)
        self.attention2 = attention2 or Attention2(256)  # New attention layer
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(256), nn.Dropout(), nn.Linear(256, n_out)
        )

    def forward(self, bags, lens):
        embeddings = self.encoder(bags)
        masked_attention_scores = self._masked_attention_scores(embeddings, lens)

        # Apply the second attention layer
        masked_attention_scores2 = self._masked_attention_scores2(
            embeddings, masked_attention_scores, lens
        )
        weighted_embedding_sums = (masked_attention_scores2 * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)
        return scores
    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.

        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask, attention_scores, torch.full_like(attention_scores, -1e10)
        )
        return torch.softmax(masked_attention, dim=1)
    # Add a new function to compute the masked attention scores for the second attention layer
    def _masked_attention_scores2(self, embeddings, masked_attention_scores, lens):
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores2 = self.attention2(embeddings)

        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores2.device)
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention2 = torch.where(
            attention_mask, attention_scores2, torch.full_like(attention_scores2, -1e10)
        )
        return torch.softmax(masked_attention2 * masked_attention_scores, dim=1)  # Combine the two attention scores


def Attention2(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    n_latent = n_latent or (n_in + 1) // 2
    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))


