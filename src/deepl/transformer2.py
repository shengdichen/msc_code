import torch
import torch.nn as nn


# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        # Split the input into multiple heads
        Q = Q.view(Q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(
            1, 2
        )
        K = K.view(K.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(
            1, 2
        )
        V = V.view(V.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(
            1, 2
        )

        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model // self.n_heads) ** 0.5
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model)

        return self.out(output)


# Position-wise Feedforward Layer
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention layer
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Position-wise feedforward layer
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        positions = torch.arange(0, x.size(1)).expand(x.size(0), x.size(1)).to(x.device)
        x = self.embedding(x) + self.positional_encoding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


# Example Usage
vocab_size = 10000  # Size of the vocabulary
d_model = 512  # Dimension of the model
n_heads = 8  # Number of attention heads
n_layers = 6  # Number of encoder layers
d_ff = 2048  # Dimension of the feedforward layer
max_len = 100  # Maximum sequence length

# Instantiate the transformer encoder
transformer_encoder = TransformerEncoder(
    vocab_size, d_model, n_heads, n_layers, d_ff, max_len
)

# Test with a dummy input
dummy_input = torch.randint(
    0, vocab_size, (64, 50)
)  # Batch size of 64, sequence length of 50
output = transformer_encoder(dummy_input)

print("Output shape:", output.shape)
