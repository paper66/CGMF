import scipy.sparse as sp
from .ode_solver import *
from .cde_solver import *
from .layers import *


class CGMF(nn.Module):
    def __init__(self, device, batch_size, feature_dim, window, d_inner, n_head, d_k, d_v, kernel_sizes,
                 channel, n_mlplayer, embed_dim, dropout=0.1):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.feature_dim = feature_dim  # variables num

        self.window = window

        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.kernel_sizes = kernel_sizes
        self.channel = channel

        self.n_mlplayer = n_mlplayer

        self.embed_dim = embed_dim

        self.embed_dims = []
        for i in range(self.n_mlplayer):
            self.embed_dims.append(self.embed_dim)

        self.build_model()

    def build_model(self):

        self.convs, self.build_graph, self.ffn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.norm_c = nn.LayerNorm(self.channel)

        for kernel_size in self.kernel_sizes:
            self.convs.append(
                nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=(1, kernel_size),
                          stride=(1, kernel_size), bias=True)
            )

            self.build_graph.append(
                TransformerBlock(d_model=self.channel, d_inner=self.d_inner, n_head=self.n_head,
                                 d_k=self.d_k, d_v=self.d_v, dropout=self.dropout)
            )

            self.ffn.append(
                nn.Sequential(
                    nn.Linear(self.channel, self.channel),
                    nn.PReLU(),
                    nn.LayerNorm(self.channel)
                )
            )

        # ODE-GNN
        self.edge_solver = DiffeqSolver(ODEGraphFunc(), "euler", odeint_rtol=1e-3, odeint_atol=1e-4,
                                        device=self.device).to(self.device)

        # CDE
        len_s1, len_s2, len_s3 = self.window // self.kernel_sizes[0], self.window // self.kernel_sizes[
            1], self.window // self.kernel_sizes[2]
        scale_num = len_s1 + len_s2 + len_s3

        self.gru = nn.GRU(self.feature_dim, self.channel * self.feature_dim, batch_first=True, dropout=self.dropout,
                          bidirectional=False)
        self.cde = CDESolver(self.channel, self.channel, self.d_inner, self.n_head, self.d_k, self.d_v,
                             interpolation="cubic")

        ''' output '''
        self.out_mlp = MultiLayerPerceptron(self.channel * (scale_num + len_s1), self.embed_dims, self.dropout,
                                            output_layer=True)

    def forward(self, X):
        X0 = X[:, -1, :]
        X = X.permute(0, 2, 1).unsqueeze(1)

        ''' Multi-scale Temporal Embedding '''
        s1, s2, s3 = self.convs[0](X), self.convs[1](X), self.convs[2](X)
        s1, s2, s3 = s1.permute(0, 2, 3, 1), s2.permute(0, 2, 3, 1), s3.permute(0, 2, 3, 1)
        # s.shape: [batch, variable, length, hidden]
        s1, s2, s3 = nn.Sigmoid()(s1), nn.Sigmoid()(s2), nn.Sigmoid()(s3)

        ''' Dynamic Temporal Graph '''
        g1 = s1.reshape((s1.shape[0], -1, s1.shape[-1]))  # g.shape [batch, time*variables, hidden_size]
        g2 = s2.reshape((s2.shape[0], -1, s2.shape[-1]))
        g3 = s3.reshape((s3.shape[0], -1, s3.shape[-1]))

        _, attn_s1 = self.build_graph[0](g1, g1)
        _, attn_s2 = self.build_graph[1](g2, g2)
        _, attn_s3 = self.build_graph[2](g3, g3)

        attn_s1, attn_s2, attn_s3 = attn_s1.sum(dim=1) / self.n_head, attn_s2.sum(dim=1) / self.n_head, \
                                    attn_s3.sum(dim=1) / self.n_head

        ''' Long-range Intra- and Inter-Relations Learning '''
        gen_timestep = torch.linspace(0, 2, steps=40, device=self.device)

        self.edge_solver.ode_func.A1 = adj_normalize(attn_s1, self_loop=True, symmetric=False)
        self.edge_solver.ode_func.A2 = adj_normalize(attn_s2, self_loop=True, symmetric=False)
        self.edge_solver.ode_func.A3 = adj_normalize(attn_s3, self_loop=True, symmetric=False)

        self.edge_solver.ode_func.mode = "A1"
        g1_ode = self.edge_solver(g1, gen_timestep)[-1]

        self.edge_solver.ode_func.mode = "A2"
        g2_ode = self.edge_solver(g2, gen_timestep)[-1]

        self.edge_solver.ode_func.mode = "A3"
        g3_ode = self.edge_solver(g3, gen_timestep)[-1]

        g1, g2, g3 = self.ffn[0](g1_ode) + g1, self.ffn[1](g2_ode) + g2, self.ffn[2](g3_ode) + g3

        s1 = g1.reshape((s1.shape[0], self.feature_dim, s1.shape[2], s1.shape[3]))
        s2 = g2.reshape((s2.shape[0], self.feature_dim, s2.shape[2], s2.shape[3]))
        s3 = g3.reshape((s3.shape[0], self.feature_dim, s3.shape[2], s3.shape[3]))

        ''' Coress-scale Representation Fusion '''
        zt, z0 = self.gru(X.reshape((-1, self.window, self.feature_dim)).flip(dims=[1]))
        z0 = z0.reshape((s1.shape[0] * self.feature_dim, -1))
        zt = zt[:, torch.linspace(0, self.window - 24, 8).long(), :].flip(dims=[0]) \
            .reshape((s1.shape[0] * self.feature_dim, s1.shape[2], -1))

        c1 = s1.reshape((s1.shape[0] * self.feature_dim, s1.shape[2], -1))
        c2 = s2.repeat(1, 1, 1, 2).reshape((c1.shape[0], c1.shape[1], -1))
        c3 = s3.repeat(1, 1, 1, 4).reshape((c1.shape[0], c1.shape[1], -1))

        time_step = torch.linspace(0, 8 - 1, 8*2, device=c1.device)
        self.cde.func.c = torch.stack((c1, c2, c3), dim=2)
        e = self.norm_c(self.cde(z0, zt, time_step))[:,torch.linspace(4-1, 8*2-1, 8).long()]
        s = torch.cat((s1, s2, s3), dim=2)

        s = s.reshape((s1.shape[0], self.feature_dim, -1))
        e = e.reshape((s1.shape[0], self.feature_dim, -1))

        o = self.out_mlp((torch.cat((s, torch.relu(e)), dim=-1)))

        return (o + X0) / 2


def adj_normalize(A, self_loop=False, symmetric=True):
    # A_normed = D^(-0.5)(A+I)D^(-0.5)
    if self_loop:
        A = A + torch.eye(A.shape[1], device=A.device).repeat(A.shape[0], 1, 1)
    degree = A.sum(dim=2)
    if symmetric:
        D = torch.diag_embed(torch.pow(degree, -0.5))
        return D.bmm(A).bmm(D)
    else:
        D = torch.diag_embed(torch.pow(degree, -1))
        return torch.bmm(D, A)