import torch


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.glu(x)

class Glu(torch.nn.Module):
    
    def __init__(self, dim):
        super(Glu, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()