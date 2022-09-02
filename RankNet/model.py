import torch.nn as nn

class RankNet(nn.Module):
    def __init__(self, inpt_dim, p_dropout=0.2):
        super(RankNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(inpt_dim, 256),
            nn.Dropout(p_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.Dropout(p_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
