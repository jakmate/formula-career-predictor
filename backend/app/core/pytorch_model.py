import torch.nn as nn


class RacingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.3):
        super().__init__()
        # Single hidden layer
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize weights for stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze(-1)
