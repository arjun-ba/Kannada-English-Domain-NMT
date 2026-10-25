"""
Microservice: Translation Service - Seq2Seq Architecture
Encoder-Decoder Neural Network with Bahdanau Attention Mechanism.
"""

import random
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from .attention import BahdanauAttention

class Encoder(nn.Module):
    """Bidirectional GRU Encoder."""

    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [src_len, batch_size]
        Returns:
            outputs: [src_len, batch_size, hidden_dim * 2]
            hidden: [batch_size, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # hidden is [2, batch_size, hidden_dim]
        # Concat forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_final = torch.tanh(self.fc(hidden_cat))

        return outputs, hidden_final


class Decoder(nn.Module):
    """GRU Decoder with Bahdanau Attention."""

    def __init__(self, output_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.attention = BahdanauAttention(hidden_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hidden_dim * 2) + emb_dim, hidden_dim)
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_token: [batch_size]
            hidden: [batch_size, hidden_dim]
            encoder_outputs: [src_len, batch_size, hidden_dim * 2]
        Returns:
            prediction: [batch_size, output_dim]
            hidden: [batch_size, hidden_dim]
            attention_weights: [batch_size, src_len]
        """
        input_token = input_token.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input_token))  # [1, batch_size, emb_dim]

        # Calculate attention weights and context vector
        context, weights = self.attention(hidden, encoder_outputs)

        # Combine embedded input and context
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        hidden = hidden.squeeze(0)
        output = output.squeeze(0)
        embedded = embedded.squeeze(0)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return prediction, hidden, weights


class Seq2Seq(nn.Module):
    """Full End-to-End Sequence-to-Sequence NMT Model."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass during training.
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[0, :]

        for t in range(1, trg_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[t] if teacher_force else top1

        return outputs
