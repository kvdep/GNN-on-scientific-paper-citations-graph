import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)


class TopologicalHeuristicsCalculator:
    """Computes pairwise topological and contextual link heuristics.

    Includes publication year difference, Jaccard overlap of co-authors,
    concept similarity, log-transformed target in-degree (hubness prior),
    Common Neighbors, and Adamic-Adar.
    """

    STOP_CONCEPTS = {"artificial", "intelligence", "machine", "learning", "computer", "science"}

    def __init__(
        self,
        ordered_papers: List[str],
        year_dict: Dict[str, int],
        authors_dict: Dict[str, Set[str]],
        concepts_dict: Dict[str, Set[str]],
        in_degree_dict: Dict[str, int],
    ) -> None:
        self.ordered_papers = ordered_papers
        self.num_papers = len(ordered_papers)
        self.paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}

        self.years_array = np.array(
            [year_dict.get(p, 0) for p in ordered_papers], dtype=np.int32
        )
        self.in_degree_log_array = np.array(
            [np.log(in_degree_dict.get(p, 0) + 1.0) for p in ordered_papers],
            dtype=np.float32,
        )

        # Build sparse author-paper incidence matrix
        all_authors = sorted(
            list(set(a for p in ordered_papers for a in authors_dict.get(p, set())))
        )
        self.author_to_idx = {a: i for i, a in enumerate(all_authors)}
        r_a, c_a = [], []
        for p in ordered_papers:
            pidx = self.paper_to_idx[p]
            for a in authors_dict.get(p, set()):
                r_a.append(pidx)
                c_a.append(self.author_to_idx[a])
        self.M_A = sp.csr_matrix(
            (np.ones(len(r_a), dtype=np.float32), (r_a, c_a)),
            shape=(self.num_papers, max(len(all_authors), 1)),
        )
        self.deg_A = np.array(self.M_A.sum(axis=1)).flatten()

        # Build sparse concept-paper incidence matrix (excluding stop words)
        all_concepts = sorted(
            list(
                set(
                    c
                    for p in ordered_papers
                    for c in concepts_dict.get(p, set())
                    if c not in self.STOP_CONCEPTS
                )
            )
        )
        self.concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
        r_c, c_c = [], []
        for p in ordered_papers:
            pidx = self.paper_to_idx[p]
            for c in concepts_dict.get(p, set()):
                if c in self.concept_to_idx:
                    r_c.append(pidx)
                    c_c.append(self.concept_to_idx[c])
        self.M_C = sp.csr_matrix(
            (np.ones(len(r_c), dtype=np.float32), (r_c, c_c)),
            shape=(self.num_papers, max(len(all_concepts), 1)),
        )
        self.deg_C = np.array(self.M_C.sum(axis=1)).flatten()

    def get_heuristics_batch(
        self,
        src_indices: np.ndarray,
        dst_indices: Optional[np.ndarray] = None,
        is_full: bool = False,
        include_hub: bool = True,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Vectorized computation of pair heuristics.

        Parameters
        ----------
        src_indices : np.ndarray
            Array of source node indices (queries).
        dst_indices : Optional[np.ndarray]
            Array of destination node indices (candidates). Required if is_full=False.
        is_full : bool
            If True, calculates Cartesian product: src_indices against all nodes in the graph.
        include_hub : bool
            Whether to append the in-degree hubness feature as the 4th dimension.
        device : str
            Target PyTorch device ('cuda' or 'cpu').

        Returns
        -------
        torch.Tensor of shape (batch_size, num_features) or (batch_size, num_papers, num_features)
        """
        B = len(src_indices)
        if is_full:
            # 1-vs-all Cartesian mode
            dt = self.years_array[src_indices].reshape(-1, 1) - self.years_array.reshape(1, -1)
            hub = np.repeat(self.in_degree_log_array.reshape(1, -1), B, axis=0)

            # Author Jaccard: I_A / (deg(u) + deg(v) - I_A)
            I_A = self.M_A[src_indices].dot(self.M_A.T).toarray()
            U_A = self.deg_A[src_indices].reshape(-1, 1) + self.deg_A.reshape(1, -1) - I_A
            au_o = np.where(U_A > 0, I_A / U_A, 0.0)

            # Concept Jaccard: I_C / (deg(u) + deg(v) - I_C)
            I_C = self.M_C[src_indices].dot(self.M_C.T).toarray()
            U_C = self.deg_C[src_indices].reshape(-1, 1) + self.deg_C.reshape(1, -1) - I_C
            cn_o = np.where(U_C > 0, I_C / U_C, 0.0)
        else:
            if dst_indices is None:
                raise ValueError("dst_indices must be provided when is_full=False")
            dt = (self.years_array[src_indices] - self.years_array[dst_indices]).reshape(-1, 1)
            hub = self.in_degree_log_array[dst_indices].reshape(-1, 1)

            # Dot product of pairs
            I_A = self.M_A[src_indices].multiply(self.M_A[dst_indices]).sum(axis=1).A1.reshape(-1, 1)
            U_A = self.deg_A[src_indices].reshape(-1, 1) + self.deg_A[dst_indices].reshape(-1, 1) - I_A
            au_o = np.where(U_A > 0, I_A / U_A, 0.0)

            I_C = self.M_C[src_indices].multiply(self.M_C[dst_indices]).sum(axis=1).A1.reshape(-1, 1)
            U_C = self.deg_C[src_indices].reshape(-1, 1) + self.deg_C[dst_indices].reshape(-1, 1) - I_C
            cn_o = np.where(U_C > 0, I_C / U_C, 0.0)

        features = [dt, au_o, cn_o]
        if include_hub:
            features.append(hub)

        stacked = np.stack(features, axis=-1).astype(np.float32)
        dev = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        return torch.from_numpy(stacked).to(dev)
