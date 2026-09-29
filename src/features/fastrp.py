import logging
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)


class FastRP:
    """Fast Random Projection (FastRP) for scalable network embedding.

    Computes multi-hop neighborhood random projections on sparse transition
    matrices without gradient backpropagation.
    """

    def __init__(
        self,
        dim: int = 128,
        num_steps: int = 3,
        weights: Optional[List[float]] = None,
        random_seed: int = 42,
    ) -> None:
        self.dim = dim
        self.num_steps = num_steps
        self.weights = weights or [0.1, 0.4, 0.5]
        self.random_seed = random_seed

    def compute(
        self,
        graph: nx.DiGraph,
        ordered_papers: List[str],
    ) -> torch.Tensor:
        """Compute FastRP embeddings for all papers in ordered_papers.

        Constructs symmetric projection across citations and co-authorships,
        generates the sparse random walk transition matrix, and computes projections.
        """
        logger.info(
            f"Computing FastRP (dim={self.dim}, steps={self.num_steps}) for {len(ordered_papers)} papers..."
        )
        paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}
        num_nodes = len(ordered_papers)

        # 1. Collect undirected paper-paper citation edges within ordered_papers
        rows: List[int] = []
        cols: List[int] = []
        for u, v, d in graph.edges(data=True):
            if d.get("type") == "cites" and u in paper_to_idx and v in paper_to_idx:
                ui = paper_to_idx[u]
                vi = paper_to_idx[v]
                rows.extend([ui, vi])
                cols.extend([vi, ui])

        # 2. Add bipartite co-authorship projection (papers sharing at least one author)
        author_to_papers: Dict[str, List[int]] = {}
        for p in ordered_papers:
            pidx = paper_to_idx[p]
            for neighbor in graph.neighbors(p):
                if graph.nodes[neighbor].get("type") == "author":
                    author_to_papers.setdefault(neighbor, []).append(pidx)

        for author, papers in author_to_papers.items():
            if len(papers) > 1:
                for i in range(len(papers)):
                    for j in range(i + 1, min(i + 10, len(papers))):
                        rows.extend([papers[i], papers[j]])
                        cols.extend([papers[j], papers[i]])

        # 3. Create sparse adjacency matrix
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        adj.sum_duplicates()

        # Add self-loops to prevent zero degrees
        adj = adj + sp.eye(num_nodes, dtype=np.float32)

        # 4. Form random walk transition matrix P = D^-1 * A
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.power(deg, -1.0)
        deg_inv[np.isinf(deg_inv)] = 0.0
        d_mat = sp.diags(deg_inv)
        p_trans = d_mat.dot(adj)

        # 5. Generate random projection matrix R via Achlioptas sparse distribution
        rng = np.random.default_rng(self.random_seed)
        r_mat = rng.standard_normal((num_nodes, self.dim)).astype(np.float32)
        r_mat /= np.sqrt(self.dim)

        # 6. Multi-step aggregation: Z = sum_{l=1}^L w_l * P^l * R
        current_p = r_mat.copy()
        z_embed = np.zeros_like(r_mat)

        for step, w in enumerate(self.weights, start=1):
            current_p = p_trans.dot(current_p)
            z_embed += w * current_p
            logger.info(f"FastRP step {step}/{self.num_steps} complete.")

        # L2-normalization
        norms = np.linalg.norm(z_embed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z_embed = z_embed / norms

        logger.info(f"FastRP completed. Shape: {z_embed.shape}")
        return torch.from_numpy(z_embed)

    @staticmethod
    def aggregate_author_topology(
        ordered_papers: List[str],
        authors_dict: Dict[str, Set[str]],
        paper_topo_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute author topological embeddings and project onto papers via mean pooling.

        For each author, average the FastRP embeddings of their papers.
        Then, for each paper, average the embeddings of its authors.
        """
        logger.info("Aggregating author topology from paper FastRP embeddings...")
        dim = paper_topo_embeddings.shape[1]
        paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}

        author_to_paper_indices: Dict[str, List[int]] = {}
        for p in ordered_papers:
            pidx = paper_to_idx[p]
            for a in authors_dict.get(p, set()):
                author_to_paper_indices.setdefault(a, []).append(pidx)

        # Author embeddings = mean of papers
        author_embeddings: Dict[str, np.ndarray] = {}
        topo_np = paper_topo_embeddings.numpy()
        for a, p_indices in author_to_paper_indices.items():
            author_embeddings[a] = np.mean(topo_np[p_indices], axis=0)

        # Paper author topology = mean of author embeddings
        aggregated = np.zeros((len(ordered_papers), dim), dtype=np.float32)
        for i, p in enumerate(ordered_papers):
            authors = authors_dict.get(p, set())
            valid_vecs = [author_embeddings[a] for a in authors if a in author_embeddings]
            if valid_vecs:
                aggregated[i] = np.mean(valid_vecs, axis=0)

        # Normalize
        norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        aggregated /= norms

        return torch.from_numpy(aggregated)
