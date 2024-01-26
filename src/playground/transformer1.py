import torch
import torch.nn as nn


# Transformer Model
class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers
    ):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        output = self.transformer(src, tgt)
        output = self.fc(output)

        return output


# Hyperparameters
vocab_size = 10000  # Vocabulary size
d_model = 512  # Model dimensionality
nhead = 8  # Number of attention heads
num_encoder_layers = 6  # Number of encoder layers
num_decoder_layers = 6  # Number of decoder layers

# Initialize the model
model = Transformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)

# Example input
src = torch.randint(0, vocab_size, (10, 32))  # Source sequence
tgt = torch.randint(0, vocab_size, (20, 32))  # Target sequence

# Forward pass
output = model(src, tgt)

# Print the output shape
print("Output Shape:", output.shape)
