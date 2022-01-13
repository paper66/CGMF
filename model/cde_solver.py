import torch
import torchcde
from .layers import *


# input: [H_1, H_2, ..., H_t], t is a variable
# output: [H_1, H_2, ..., H_n], n is fixed
class CDESolver(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, d_inner, n_head, d_k, d_v, interpolation="cubic"):
        super(CDESolver, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, d_inner, n_head, d_k, d_v)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.interpolation = interpolation

    def forward(self, z0, z, t):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(z)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        adjoint_params = tuple(self.func.parameters()) + (coeffs,)

        z0 = self.initial(z0)
        e_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=t, adjoint_params=adjoint_params, method="euler")
        return e_T


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, d_inner, n_head, d_k, d_v):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.c = None

        self.cs_attn = TransformerBlock(d_model=input_channels, d_inner=d_inner,
                                        n_head=n_head, d_k=d_k, d_v=d_v)

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, input_channels)

    def forward(self, t, e):
        attn = torch.cat((e.unsqueeze(dim=1), self.c[:, int(t)]), dim=1)
        e_attn, _ = self.cs_attn(attn, attn)
        e_attn = e_attn[:, 0]
        e_hidden = torch.relu(self.linear1(e_attn))
        e_input = torch.relu(self.linear2(e_attn))
        e = torch.tanh(torch.bmm(e_hidden.unsqueeze(dim=2), e_input.unsqueeze(dim=1)))
        e = e.view(e.size(0), self.hidden_channels, self.input_channels)
        return e
