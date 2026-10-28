"""Edge-classification GNNs on (sampled) directed multigraphs.

arch
  hetero   Multi-GNN semantics: PyG ``to_hetero(model, aggr='mean')`` over relations
           ``to`` / ``rev_to``. Per layer h <- (h + ReLU(BN(mean(conv_to, conv_rev)))) / 2;
           each relation keeps its own edge state, updated by its own edge MLP
           (Egressy et al., AAAI 2024; github.com/IBM/Multi-GNN, Apache-2.0).
  dual     SymDiG: forward stream on G_S^+ and reverse stream on G_S^- (its exact flip)
           propagate independently for all L layers, each with its own node AND edge
           state (no cross-stream channel), and are fused only at the end
           (fusion = concat | mha).
  dir      Dir-GNN-style baseline (Rossi et al., 2023): per layer
           alpha_l * conv_out + (1 - alpha_l) * conv_in with a learnable alpha_l.
  egsage   E-GraphSAGE (Lo et al., 2022) on IP:PORT nodes: h_v <- ReLU(W [h_v || mean of
           incident edge features]), edge features not updated, readout [z_u || z_v].

Readout for hetero/dual/dir: MLP([ReLU([h_u || h_v]) || edge state]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.utils import scatter


def make_conv(kind, D, heads, dropout):
    if kind == "gine":
        return GINEConv(nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)), edge_dim=D)
    if kind == "gatv2":
        return GATv2Conv(D, D // heads, heads=heads, edge_dim=D, add_self_loops=False, dropout=dropout)
    raise ValueError(kind)


def edge_mlp(n_in, D):
    return nn.Sequential(nn.Linear(n_in * D, D), nn.ReLU(), nn.Linear(D, D))


def readout(n_in, D, n_classes, dropout):
    return nn.Sequential(nn.Linear(n_in, D), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(D, D // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(D // 2, n_classes))


class EdgeGNN(nn.Module):
    def __init__(self, d_edge, n_classes, arch="dual", conv="gine", hidden=128, layers=2, heads=4,
                 dropout=0.2, fusion="concat", ego=True, edge_updates=True):
        super().__init__()
        D = (hidden // heads) * heads if conv == "gatv2" else hidden
        self.arch, self.fusion, self.ego, self.edge_updates, self.L, self.D = arch, fusion, ego, edge_updates, layers, D
        self.node_enc = nn.Linear(2 if ego else 1, D)
        self.edge_enc = nn.Linear(d_edge, D)
        self.drop = nn.Dropout(dropout)
        stack = lambda f: nn.ModuleList([f() for _ in range(layers)])
        if arch in ("hetero", "dir", "dual"):
            self.conv_f = stack(lambda: make_conv(conv, D, heads, dropout))
            self.conv_r = stack(lambda: make_conv(conv, D, heads, dropout))
        if arch in ("hetero", "dir"):
            self.bn = stack(lambda: nn.BatchNorm1d(D))
            if edge_updates:
                self.emlp_f = stack(lambda: edge_mlp(3, D))
                self.emlp_r = stack(lambda: edge_mlp(3, D))
            if arch == "dir":
                self.alpha = nn.Parameter(torch.zeros(layers))
            self.cls = readout(3 * D, D, n_classes, dropout)
        elif arch == "dual":
            self.bn_f = stack(lambda: nn.BatchNorm1d(D))
            self.bn_r = stack(lambda: nn.BatchNorm1d(D))
            if edge_updates:
                self.emlp_f = stack(lambda: edge_mlp(3, D))
                self.emlp_r = stack(lambda: edge_mlp(3, D))
            if fusion == "mha":
                self.mha = nn.MultiheadAttention(D, heads, batch_first=True)
            self.cls = readout(6 * D, D, n_classes, dropout)
        elif arch == "egsage":
            self.sage = stack(lambda: nn.Linear(2 * D, D))
            self.cls = readout(2 * D, D, n_classes, dropout)
        else:
            raise ValueError(arch)

    def _x0(self, b):
        x = b["x_node"] if self.ego else b["x_node"][:, :1]
        return self.node_enc(x)

    def forward(self, b):
        h = self._x0(b)
        ei_f = b["ei_f"]
        sf, df = ei_f[0], ei_f[1]
        tp = b["tpos"]
        st, dt = sf[tp], df[tp]
        e_f = self.edge_enc(b["xf"])
        out = {}
        if self.arch == "egsage":
            n = h.size(0)
            idx = torch.cat([sf, df])
            agg = scatter(torch.cat([e_f, e_f]), idx, dim=0, dim_size=n, reduce="mean")
            for l in range(self.L):
                h = self.drop(F.relu(self.sage[l](torch.cat([h, agg], 1))))
            z = torch.cat([h[st], h[dt]], 1)
            out["logits"] = self.cls(z)
            out["h"] = h
            return out
        if self.arch == "dual":
            if not b["shared"]:
                raise ValueError("arch=dual needs the coupled sampler (shared edge set)")
            ei_r = ei_f.flip(0)
            hp, hm, ep, em = h, h, e_f, e_f
            for l in range(self.L):
                mp = self.conv_f[l](hp, ei_f, ep)
                mm = self.conv_r[l](hm, ei_r, em)
                hp = 0.5 * (hp + self.drop(F.relu(self.bn_f[l](mp))))
                hm = 0.5 * (hm + self.drop(F.relu(self.bn_r[l](mm))))
                if self.edge_updates:
                    ep = ep + self.emlp_f[l](torch.cat([hp[sf], hp[df], ep], 1)) / 2
                    em = em + self.emlp_r[l](torch.cat([hm[df], hm[sf], em], 1)) / 2
            if self.fusion == "mha":
                A = torch.stack([hp, hm], 1)
                fused, _ = self.mha(A, A, A, need_weights=False)
                hf = fused.reshape(hp.size(0), 2 * self.D)
            else:
                hf = torch.cat([hp, hm], 1)
            z = torch.cat([F.relu(torch.cat([hf[st], hf[dt]], 1)), ep[tp], em[tp]], 1)
            out.update(logits=self.cls(z), h=hf)
            return out
        # hetero / dir
        ei_r = b["ei_r"]
        sr, dr = ei_r[0], ei_r[1]
        e_r = e_f if b["shared"] else self.edge_enc(b["xr"])
        for l in range(self.L):
            m_f = self.conv_f[l](h, ei_f, e_f)
            m_r = self.conv_r[l](h, ei_r, e_r)
            if self.arch == "dir":
                a = torch.sigmoid(self.alpha[l])
                m = a * m_r + (1 - a) * m_f
            else:
                m = 0.5 * (m_f + m_r)
            h = 0.5 * (h + self.drop(F.relu(self.bn[l](m))))
            if self.edge_updates:
                e_f_new = e_f + self.emlp_f[l](torch.cat([h[sf], h[df], e_f], 1)) / 2
                e_r = e_r + self.emlp_r[l](torch.cat([h[sr], h[dr], e_r], 1)) / 2
                e_f = e_f_new
        z = torch.cat([F.relu(torch.cat([h[st], h[dt]], 1)), e_f[tp]], 1)
        out.update(logits=self.cls(z), h=h)
        return out


def build_model(mcfg, d_edge, n_classes):
    return EdgeGNN(d_edge, n_classes, arch=mcfg["arch"], conv=mcfg.get("conv", "gine"),
                   hidden=int(mcfg.get("hidden", 128)), layers=int(mcfg.get("layers", 2)),
                   heads=int(mcfg.get("heads", 4)), dropout=float(mcfg.get("dropout", 0.2)),
                   fusion=mcfg.get("fusion", "concat"), ego=bool(mcfg.get("ego", True)),
                   edge_updates=bool(mcfg.get("edge_updates", True)))
