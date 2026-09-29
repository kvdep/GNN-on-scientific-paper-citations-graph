import logging
from typing import Dict, List, Optional, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CausalNegativeSampler:
    """Samples negative edges respecting temporal causality and semantic hardness.

    Ensures that publication year y(v_neg) <= y(u), eliminating look-ahead bias
    and time-travel leakage in citation link prediction.
    """

    def __init__(self, years: np.ndarray, random_seed: int = 42) -> None:
        self.years = years
        self.num_nodes = len(years)
        self.rng = np.random.default_rng(random_seed)

    def sample_causal_negatives(
        self,
        src_indices: np.ndarray,
        max_attempts: int = 10,
    ) -> np.ndarray:
        """Sample negative destinations v_neg such that year(v_neg) <= year(u).

        Parameters
        ----------
        src_indices : np.ndarray
            Array of source paper indices u.
        max_attempts : int
            Maximum number of resampling attempts for invalid candidates.

        Returns
        -------
        np.ndarray
            Array of sampled negative destination indices of same shape as src_indices.
        """
        src_years = self.years[src_indices]
        neg_candidates = self.rng.integers(0, self.num_nodes, size=len(src_indices))

        invalid_mask = (self.years[neg_candidates] > src_years) | (neg_candidates == src_indices)
        attempts = 0

        while np.any(invalid_mask) and attempts < max_attempts:
            num_invalid = np.sum(invalid_mask)
            resampled = self.rng.integers(0, self.num_nodes, size=num_invalid)
            neg_candidates[invalid_mask] = resampled
            invalid_mask = (self.years[neg_candidates] > src_years) | (neg_candidates == src_indices)
            attempts += 1

        # Fallback: if any still invalid, pick min year node
        if np.any(invalid_mask):
            min_year_idx = int(np.argmin(self.years))
            neg_candidates[invalid_mask] = min_year_idx

        return neg_candidates

    @staticmethod
    def sample_hard_negatives(
        src_indices: np.ndarray,
        positive_edges: np.ndarray,
        text_embeddings: torch.Tensor,
        years: np.ndarray,
        top_k: int = 50,
        device: str = "cuda",
    ) -> np.ndarray:
        """Select hard negatives based on high semantic cosine similarity.

        Finds candidate nodes that have high SciBERT similarity to u, are causally valid,
        but are not true citations of u.
        """
        logger.info(f"Generating hard negatives for {len(src_indices)} edges (top_k={top_k})...")
        num_nodes = text_embeddings.size(0)
        norm_embs = torch.nn.functional.normalize(text_embeddings, p=2, dim=1).to(device)
        years_t = torch.tensor(years, device=device)

        true_neighbors: Dict[int, Set[int]] = {}
        for u, v in positive_edges:
            true_neighbors.setdefault(int(u), set()).add(int(v))

        hard_negatives = np.empty(len(src_indices), dtype=np.int64)

        # Batch queries
        batch_size = 512
        for i in range(0, len(src_indices), batch_size):
            batch_src = src_indices[i : i + batch_size]
            b_u_t = torch.tensor(batch_src, device=device)
            b_years = years_t[b_u_t]

            # Compute similarity
            sim = torch.mm(norm_embs[b_u_t], norm_embs.t())
            # Mask lookahead
            sim[years_t.unsqueeze(0) > b_years.unsqueeze(1)] = -2.0
            # Mask self
            for j, u_val in enumerate(batch_src):
                sim[j, u_val] = -2.0

            _, top_candidates = torch.topk(sim, min(top_k, num_nodes), dim=1)
            top_candidates_cpu = top_candidates.cpu().numpy()

            for j, u_val in enumerate(batch_src):
                cands = top_candidates_cpu[j]
                u_gt = true_neighbors.get(int(u_val), set())
                selected = -1
                for c in cands:
                    if int(c) not in u_gt:
                        selected = int(c)
                        break
                if selected == -1:
                    selected = int(cands[0])
                hard_negatives[i + j] = selected

        return hard_negatives
