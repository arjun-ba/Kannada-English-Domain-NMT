"""
Microservice: Translation Service - Attention Mechanism
Implements Bahdanau (Additive) Attention for sequence-to-sequence Neural Machine Translation.
Calculates attention alignment weights between source Kannada tokens and target English tokens.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    Computes alignment score:
    energy = v^T * tanh(W_h * hidden + W_e * encoder_outputs)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_encoder = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # Bidirectional encoder (* 2)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch_size, hidden_dim] (decoder hidden state)
            encoder_outputs: [src_len, batch_size, hidden_dim * 2]
        Returns:
            context: [batch_size, hidden_dim * 2]
            attention_weights: [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        # hidden: [batch_size, hidden_dim] -> [batch_size, src_len, hidden_dim]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs: [src_len, batch_size, hidden_dim * 2] -> [batch_size, src_len, hidden_dim * 2]
        encoder_outputs_transposed = encoder_outputs.permute(1, 0, 2)

        # Calculate energy
        energy = torch.tanh(self.w_hidden(hidden_expanded) + self.w_encoder(encoder_outputs_transposed))
        # attention: [batch_size, src_len, 1] -> [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        weights = F.softmax(attention, dim=1)

        # Calculate context vector
        # weights: [batch_size, 1, src_len]
        # encoder_outputs_transposed: [batch_size, src_len, hidden_dim * 2]
        # context: [batch_size, 1, hidden_dim * 2] -> [batch_size, hidden_dim * 2]
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs_transposed).squeeze(1)

        return context, weights
