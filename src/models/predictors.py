import torch
import torch.nn as nn
from typing import Optional

from src.models.gnn_encoders import DirLightGCN, LightGCN, NeoGNN_Encoder, SAGE, SGC


class Standard_Predictor(nn.Module):
    """Early Fusion Edge Predictor.

    Concatenates source representation, destination representation, their Hadamard product,
    absolute difference, and structural heuristics vector:
    f = [z_u || z_v || z_u * z_v || z_u - z_v || heuristics]
    """

    def __init__(self, emb_dim: int, num_heuristics: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = (emb_dim * 4) + num_heuristics
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_u: torch.Tensor, z_v: torch.Tensor, heuristics: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([z_u, z_v, z_u * z_v, z_u - z_v, heuristics], dim=-1)
        return self.mlp(feat).squeeze(-1)


class BUDDY_Predictor(nn.Module):
    """Late Fusion Edge Predictor (BUDDY-style).

    Decouples semantic vector processing from topological subgraph heuristics
    via independent dense pathways before final classification.
    """

    def __init__(self, emb_dim: int, num_heuristics: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.semantic_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.struct_mlp = nn.Sequential(
            nn.Linear(num_heuristics, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_u: torch.Tensor, z_v: torch.Tensor, heuristics: torch.Tensor
    ) -> torch.Tensor:
        sem_rep = self.semantic_mlp(torch.cat([z_u, z_v], dim=-1))
        str_rep = self.struct_mlp(heuristics)
        fused = torch.cat([sem_rep, str_rep], dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)


class NCN_Predictor(nn.Module):
    """Neighborhood Common Neighbors (NCN) Projection Predictor.

    Relies on element-wise Hadamard interaction and topological features.
    """

    def __init__(self, emb_dim: int, num_heuristics: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + num_heuristics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_u: torch.Tensor, z_v: torch.Tensor, heuristics: torch.Tensor
    ) -> torch.Tensor:
        interaction = z_u * z_v
        feat = torch.cat([interaction, heuristics], dim=-1)
        return self.mlp(feat).squeeze(-1)


class LinkCausalPredictor(nn.Module):
    """Simple linear projection predictor for causal link scoring."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(dim * 2, dim)
        self.l2 = nn.Linear(dim, 1)

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_s, z_d], dim=-1)
        x = torch.relu(self.l1(x))
        return self.l2(x).squeeze(-1)


class GM(nn.Module):
    """Graph Model (GM) Container combining Encoder and Predictor.

    Parameters
    ----------
    n_f : int
        Input feature dimension of nodes.
    n_v : int
        Total number of nodes in graph (required for NeoGNN).
    n_a : int
        Dimension of pairwise heuristics vector.
    o_d : int
        Latent embedding dimension.
    g_type : str
        Encoder type: 'none', 'lightgcn', 'dirgcn', 'neognn', 'sgc', 'sage'.
    p_type : str
        Predictor type: 'standard', 'buddy', 'ncn'.
    """

    def __init__(
        self,
        n_f: int,
        n_v: int,
        n_a: int = 4,
        o_d: int = 128,
        g_type: str = "lightgcn",
        p_type: str = "buddy",
        **kwargs,
    ) -> None:
        super().__init__()
        self.g_type = g_type.lower()
        self.p_type = p_type.lower()

        self.input_proj = nn.Linear(n_f, n_f)
        self.linear_fallback = nn.Linear(n_f, o_d)

        if self.g_type == "dirgcn":
            self.encoder = DirLightGCN(n_f, o_d)
        elif self.g_type == "sgc":
            self.encoder = SGC(n_f, o_d)
        elif self.g_type == "sage":
            self.encoder = SAGE(n_f, o_d)
        elif self.g_type == "neognn":
            self.encoder = NeoGNN_Encoder(n_v, n_f, o_d)
        elif self.g_type == "lightgcn":
            self.encoder = LightGCN(n_f, o_d)
        else:
            self.encoder = nn.Identity()

        self.norm = nn.LayerNorm(o_d)
        self.drop = nn.Dropout(0.3)

        if self.p_type == "buddy":
            self.classifier = BUDDY_Predictor(o_d, n_a)
        elif self.p_type == "ncn":
            self.classifier = NCN_Predictor(o_d, n_a)
        else:
            self.classifier = Standard_Predictor(o_d, n_a)

    def enc(
        self, x: torch.Tensor, node_indices: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode node representations."""
        h = self.input_proj(x)
        if self.g_type == "neognn":
            z = self.encoder(h, node_indices, edge_index)
        elif self.g_type != "none":
            z = self.encoder(h, edge_index)
        else:
            z = torch.relu(self.linear_fallback(h))
        return self.drop(self.norm(z))

    def prd(
        self, z: torch.Tensor, edge_pairs: torch.Tensor, heuristics: torch.Tensor
    ) -> torch.Tensor:
        """Score edge pairs."""
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]
        return self.classifier(z[u], z[v], heuristics)
