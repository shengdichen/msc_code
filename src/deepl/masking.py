import torch
import torch.nn as nn


class RNNWithMasking(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNWithMasking, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Create a mask based on sequence lengths
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None]

        # Pass through RNN with masking
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Apply the mask to ignore padding
        masked_output = output * mask.unsqueeze(2)

        # Sum along the sequence dimension to get representations
        representation = masked_output.sum(dim=1)

        # Fully connected layer
        logits = self.fc(representation)
        return logits


# Example usage
input_size = 100  # Vocabulary size
hidden_size = 128
output_size = 10  # Number of classes
max_sequence_length = 20

# Generate random input sequences of variable lengths
batch_size = 5
sequences = [
    torch.randint(0, input_size, (torch.randint(1, max_sequence_length + 1),))
    for _ in range(batch_size)
]

# Pad sequences to the maximum length
padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)

# Get lengths of each sequence
lengths = torch.tensor([len(seq) for seq in sequences])

# Create the model
model = RNNWithMasking(input_size, hidden_size, output_size)

# Forward pass with masked input
logits = model(padded_sequences, lengths)

# Print the output shape
print("Logits Shape:", logits.shape)
