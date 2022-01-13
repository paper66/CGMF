import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint

class DiffeqSolver(nn.Module):
    def __init__(self, ode_func,
                 method,
                 odeint_rtol=1e-4,
                 odeint_atol=1e-5,
                 device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        # Decode the trajectory through ODE Solver
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method).to(self.device)
        assert (torch.mean(pred_y[0] - first_point) < 0.001)

        return pred_y


class ODEGraphFunc(nn.Module):
    def __init__(self):
        super(ODEGraphFunc, self).__init__()
        self.mode = None
        self.A1 = None
        self.A2 = None
        self.A3 = None

    def forward(self, t, x):
        if self.mode == "A1":
            x = torch.bmm(self.A1, x)
        elif self.mode == "A2":
            x = torch.bmm(self.A2, x)
        elif self.mode == "A3":
            x = torch.bmm(self.A3, x)
        return x


