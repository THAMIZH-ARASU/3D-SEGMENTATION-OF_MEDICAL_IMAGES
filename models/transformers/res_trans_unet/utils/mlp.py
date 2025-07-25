from typing import Optional
import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x