import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class CatBoostCitationRanker:
    """Tabular Gradient Boosted Decision Tree (GBDT) Ranker for citation link prediction.

    Extracts a 390-dimensional feature vector per candidate pair:
    - 4 heuristic features: delta_year, author_jaccard, concept_jaccard, log_in_degree
    - 129 topological features: author FastRP cosine similarity (1d) + Hadamard product (128d)
    - 257 semantic features: SciBERT cosine similarity (1d) + Hadamard product (256d)
    Total dimension = 4 + 1 + 128 + 1 + 256 = 390.
    """

    def __init__(
        self,
        loss_function: str = "YetiRank",
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
    ) -> None:
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost is required. Install via pip install catboost"
            )
        self.loss_function = loss_function
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.model: Optional[cb.CatBoost] = None

    @staticmethod
    def extract_pair_features(
        query_indices: np.ndarray,
        candidate_indices: np.ndarray,
        text_embeddings: torch.Tensor,
        author_topology_embeddings: torch.Tensor,
        year_dict: Dict[str, int],
        authors_dict: Dict[str, Set[str]],
        concepts_dict: Dict[str, Set[str]],
        in_degree_dict: Dict[str, int],
        ordered_papers: List[str],
        stop_concepts: Optional[Set[str]] = None,
        device: str = "cuda",
    ) -> np.ndarray:
        """Vectorized extraction of the 390-dimensional feature matrix."""
        dev = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        stop_w = stop_concepts or {
            "artificial", "intelligence", "machine", "learning", "computer", "science"
        }

        N = len(query_indices)
        q_idx_t = torch.tensor(query_indices, device=dev)
        c_idx_t = torch.tensor(candidate_indices, device=dev)

        t_gpu = text_embeddings.to(dev)
        topo_gpu = author_topology_embeddings.to(dev)

        q_t = t_gpu[q_idx_t]
        c_t = t_gpu[c_idx_t]
        cos_t = torch.cosine_similarity(q_t, c_t, dim=1).unsqueeze(1)
        had_t = q_t * c_t

        q_topo = topo_gpu[q_idx_t]
        c_topo = topo_gpu[c_idx_t]
        cos_topo = torch.cosine_similarity(q_topo, c_topo, dim=1).unsqueeze(1)
        had_topo = q_topo * c_topo

        gpu_feats = torch.cat([cos_topo, had_topo, cos_t, had_t], dim=1).cpu().numpy()

        feature_matrix = np.empty((N, 390), dtype=np.float32)
        feature_matrix[:, 4:] = gpu_feats

        # Fill heuristic features
        for k in range(N):
            u = ordered_papers[query_indices[k]]
            v = ordered_papers[candidate_indices[k]]

            # Delta year
            feature_matrix[k, 0] = year_dict.get(u, 0) - year_dict.get(v, 0)

            # Author Jaccard
            au_u = authors_dict.get(u, set())
            au_v = authors_dict.get(v, set())
            union_a = len(au_u | au_v)
            feature_matrix[k, 1] = len(au_u & au_v) / union_a if union_a > 0 else 0.0

            # Concept Jaccard
            c_u = concepts_dict.get(u, set()) - stop_w
            c_v = concepts_dict.get(v, set()) - stop_w
            union_c = len(c_u | c_v)
            feature_matrix[k, 2] = len(c_u & c_v) / union_c if union_c > 0 else 0.0

            # Log target in-degree
            feature_matrix[k, 3] = np.log(in_degree_dict.get(v, 0) + 1.0)

        return feature_matrix

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        queries_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        queries_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train CatBoost ranker with group queries."""
        train_pool = cb.Pool(data=X_train, label=y_train, group_id=queries_train)
        eval_pool = None
        if X_val is not None and y_val is not None and queries_val is not None:
            eval_pool = cb.Pool(data=X_val, label=y_val, group_id=queries_val)

        self.model = cb.CatBoost(
            {
                "loss_function": self.loss_function,
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "depth": self.depth,
                "random_seed": self.random_seed,
                "verbose": 50,
                "task_type": "GPU" if torch.cuda.is_available() else "CPU",
            }
        )

        logger.info(f"Training CatBoost ranker with {self.loss_function}...")
        self.model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)
