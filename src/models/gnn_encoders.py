import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    from torch_geometric.nn import HeteroConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class LightGCN(nn.Module):
    """Light Graph Convolutional Network (LightGCN).

    Simplifies GCN by eliminating non-linear activations and weight matrices
    during neighbor aggregation, performing symmetric linear feature smoothing.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.A: Optional[torch.Tensor] = None
        self.edge_count: int = -1

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        h : torch.Tensor of shape (num_nodes, in_dim)
        edge_index : torch.Tensor of shape (2, num_edges)
        """
        z = self.w(h)
        if edge_index.size(1) == 0:
            return z

        # Construct and cache symmetric normalized adjacency matrix
        if self.training or self.A is None or self.edge_count != edge_index.size(1):
            N = h.size(0)
            u = torch.cat([edge_index[0], edge_index[1]])
            v = torch.cat([edge_index[1], edge_index[0]])
            self_loops = torch.arange(N, device=h.device)
            u = torch.cat([u, self_loops])
            v = torch.cat([v, self_loops])

            deg = torch.zeros(N, device=h.device).scatter_add_(
                0, u, torch.ones_like(u, dtype=torch.float)
            )
            d_inv_sqrt = deg.pow(-0.5)
            d_inv_sqrt.masked_fill_(d_inv_sqrt == float("inf"), 0)
            weights = d_inv_sqrt[u] * d_inv_sqrt[v]

            self.A = torch.sparse_coo_tensor(
                torch.stack([u, v]), weights, (N, N)
            ).coalesce()
            self.edge_count = edge_index.size(1)

        return torch.sparse.mm(self.A, z)


class DirLightGCN(nn.Module):
    """Directed LightGCN for citation graphs.

    Decouples incoming (cited-by) and outgoing (cites) citation flows via
    separate directed transition operators.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.A_in: Optional[torch.Tensor] = None
        self.A_out: Optional[torch.Tensor] = None
        self.edge_count: int = -1

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.w(h)
        if edge_index.size(1) == 0:
            return z

        if self.training or self.A_in is None or self.edge_count != edge_index.size(1):
            N = h.size(0)
            u, v = edge_index[0], edge_index[1]

            # Outgoing normalized adjacency
            d_out = torch.zeros(N, device=h.device).scatter_add_(
                0, u, torch.ones_like(u, dtype=torch.float)
            )
            d_out_inv = d_out.pow(-1.0)
            d_out_inv.masked_fill_(d_out_inv == float("inf"), 0)
            self.A_out = torch.sparse_coo_tensor(
                torch.stack([u, v]), d_out_inv[u], (N, N)
            ).coalesce()

            # Incoming normalized adjacency
            d_in = torch.zeros(N, device=h.device).scatter_add_(
                0, v, torch.ones_like(v, dtype=torch.float)
            )
            d_in_inv = d_in.pow(-1.0)
            d_in_inv.masked_fill_(d_in_inv == float("inf"), 0)
            self.A_in = torch.sparse_coo_tensor(
                torch.stack([v, u]), d_in_inv[v], (N, N)
            ).coalesce()
            self.edge_count = edge_index.size(1)

        z_out = torch.sparse.mm(self.A_out, z)
        z_in = torch.sparse.mm(self.A_in, z)
        return z + z_out + z_in


class NeoGNN_Encoder(nn.Module):
    """Hybrid NeoGNN Encoder.

    Processes feature convolutions through LightGCN in parallel with
    structural network topology through trainable structural embeddings.
    """

    def __init__(self, num_nodes: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.feat_gcn = LightGCN(in_dim, out_dim // 2)
        self.struct_emb = nn.Embedding(num_nodes, out_dim // 2)
        self.struct_gcn = LightGCN(out_dim // 2, out_dim // 2)

    def forward(
        self, h: torch.Tensor, node_indices: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        z_feat = self.feat_gcn(h, edge_index)

        valid_mask = (node_indices >= 0) & (
            node_indices < self.struct_emb.num_embeddings
        )
        safe_indices = torch.where(
            valid_mask, node_indices, torch.zeros_like(node_indices)
        )
        s_emb = self.struct_emb(safe_indices)
        s_emb = torch.where(valid_mask.unsqueeze(1), s_emb, torch.zeros_like(s_emb))

        z_struct = self.struct_gcn(s_emb, edge_index)
        return torch.cat([z_feat, z_struct], dim=1)


class SGC(nn.Module):
    """Simple Graph Convolution (SGC).

    Pre-computes k-hop neighborhood feature smoothing followed by linear mapping.
    """

    def __init__(self, in_dim: int, out_dim: int, k_hops: int = 2) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.k = k_hops
        self.A: Optional[torch.Tensor] = None
        self.edge_count: int = -1

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return self.w(h)

        if self.A is None or self.edge_count != edge_index.size(1):
            N = h.size(0)
            u = torch.cat([edge_index[0], edge_index[1], torch.arange(N, device=h.device)])
            v = torch.cat([edge_index[1], edge_index[0], torch.arange(N, device=h.device)])
            deg = torch.zeros(N, device=h.device).scatter_add_(
                0, u, torch.ones_like(u, dtype=torch.float)
            )
            d_inv = deg.pow(-0.5)
            d_inv.masked_fill_(d_inv == float("inf"), 0)
            weights = d_inv[u] * d_inv[v]
            self.A = torch.sparse_coo_tensor(
                torch.stack([u, v]), weights, (N, N)
            ).coalesce()
            self.edge_count = edge_index.size(1)

        x = h
        for _ in range(self.k):
            x = torch.sparse.mm(self.A, x)
        return self.w(x)


class SAGE(nn.Module):
    """GraphSAGE Layer with mean neighborhood aggregator and skip connection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)
        self.A: Optional[torch.Tensor] = None
        self.edge_count: int = -1

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return F.relu(self.w_self(h))

        if self.A is None or self.edge_count != edge_index.size(1):
            N = h.size(0)
            u = torch.cat([edge_index[0], edge_index[1]])
            v = torch.cat([edge_index[1], edge_index[0]])
            deg = torch.zeros(N, device=h.device).scatter_add_(
                0, u, torch.ones_like(u, dtype=torch.float)
            )
            d_inv = deg.pow(-1.0)
            d_inv.masked_fill_(d_inv == float("inf"), 0)
            self.A = torch.sparse_coo_tensor(
                torch.stack([u, v]), d_inv[u], (N, N)
            ).coalesce()
            self.edge_count = edge_index.size(1)

        neigh_h = torch.sparse.mm(self.A, h)
        return F.relu(self.w_self(h) + self.w_neigh(neigh_h))


class JKNet(nn.Module):
    """Jumping Knowledge Network (JKNet).

    Aggregates representations across multiple hops via max-pooling.
    """

    def __init__(self, in_dim: int, out_dim: int, k_hops: int = 3) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.k = k_hops
        self.A: Optional[torch.Tensor] = None
        self.edge_count: int = -1

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.w(h)
        if edge_index.size(1) == 0:
            return z

        if self.A is None or self.edge_count != edge_index.size(1):
            N = h.size(0)
            u = torch.cat([edge_index[0], edge_index[1], torch.arange(N, device=h.device)])
            v = torch.cat([edge_index[1], edge_index[0], torch.arange(N, device=h.device)])
            deg = torch.zeros(N, device=h.device).scatter_add_(
                0, u, torch.ones_like(u, dtype=torch.float)
            )
            d_inv = deg.pow(-0.5)
            d_inv.masked_fill_(d_inv == float("inf"), 0)
            self.A = torch.sparse_coo_tensor(
                torch.stack([u, v]), d_inv[u] * d_inv[v], (N, N)
            ).coalesce()
            self.edge_count = edge_index.size(1)

        layers = [z]
        x = z
        for _ in range(self.k):
            x = torch.sparse.mm(self.A, x)
            layers.append(x)
        return torch.stack(layers, dim=0).max(dim=0)[0]


class GAT(nn.Module):
    """Graph Attention Network (GAT) with sparse memory-efficient attention."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.a_l = nn.Linear(out_dim, 1, bias=False)
        self.a_r = nn.Linear(out_dim, 1, bias=False)
        self.a_bias = nn.Parameter(torch.zeros(1))
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.w(h)
        if edge_index.size(1) == 0:
            return z
        u = torch.cat([edge_index[0], edge_index[1]])
        v = torch.cat([edge_index[1], edge_index[0]])
        z_l = self.a_l(z)
        z_r = self.a_r(z)
        a_e = self.leaky_relu(z_l[u] + z_r[v] + self.a_bias).squeeze()
        e_w = torch.exp(a_e - a_e.max())
        s = torch.zeros(h.size(0), dtype=z.dtype, device=z.device)
        s.scatter_add_(0, u, e_w)
        a_norm = e_w / (s[u] + 1e-9)

        N = h.size(0)
        A = torch.sparse_coo_tensor(torch.stack([u, v]), a_norm, (N, N)).coalesce()
        return torch.sparse.mm(A, z) + z


class GATv2(nn.Module):
    """Dynamic Graph Attention Network (GATv2)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w_l = nn.Linear(in_dim, out_dim, bias=False)
        self.w_r = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return self.w_l(h)
        N = h.size(0)
        u = torch.cat([edge_index[0], edge_index[1], torch.arange(N, device=h.device)])
        v = torch.cat([edge_index[1], edge_index[0], torch.arange(N, device=h.device)])
        z_l = self.w_l(h)
        z_r = self.w_r(h)
        c = z_l[u] + z_r[v]
        a_e = self.a(self.leaky_relu(c)).squeeze()
        e_w = torch.exp(a_e - a_e.max())
        s = torch.zeros(N, dtype=z_l.dtype, device=z_l.device)
        s.scatter_add_(0, u, e_w)
        a_norm = e_w / (s[u] + 1e-9)
        A = torch.sparse_coo_tensor(torch.stack([u, v]), a_norm, (N, N)).coalesce()
        return torch.sparse.mm(A, z_r) + z_l


class HeteroEncoder(nn.Module):
    """Heterogeneous Graph Convolutional Encoder (HeteroGNN) using PyG."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for HeteroEncoder. Install via requirements.txt"
            )
        self.conv1 = HeteroConv(
            {
                ("author", "writes", "paper"): SAGEConv((in_dim, in_dim), hidden_dim),
                ("paper", "rev_writes", "author"): SAGEConv((in_dim, in_dim), hidden_dim),
                ("paper", "cites", "paper"): SAGEConv((in_dim, in_dim), hidden_dim),
            },
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {
                ("author", "writes", "paper"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("paper", "rev_writes", "author"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("paper", "cites", "paper"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            },
            aggr="mean",
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        return self.conv2(x_dict, edge_index_dict)
