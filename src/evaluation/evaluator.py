import logging
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from src.evaluation.metrics import calculate_all_metrics
from src.features.heuristics import TopologicalHeuristicsCalculator
from src.models.predictors import GM

logger = logging.getLogger(__name__)


class ColdStartEvaluator:
    """Evaluates link prediction and citation recommendation models in cold start.

    Emulates inductive arrival of new publications:
    - Query node u is encoded as an isolated node (no message-passing citations).
    - Candidates v are encoded with full message-passing on historical training network.
    - Causal constraint: future papers (year(v) > year(u)) and self-loops are filtered out.
    """

    def __init__(
        self,
        ordered_papers: List[str],
        year_dict: Dict[str, int],
        heuristics_calculator: TopologicalHeuristicsCalculator,
        device: str = "cuda",
    ) -> None:
        self.ordered_papers = ordered_papers
        self.num_papers = len(ordered_papers)
        self.paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}
        self.paper_array = np.array(ordered_papers)
        self.year_dict = year_dict
        self.heuristics_calculator = heuristics_calculator
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.years_gpu = torch.tensor(
            [year_dict.get(p, 0) for p in ordered_papers],
            dtype=torch.int32,
            device=self.device,
        )

    def evaluate_gnn(
        self,
        model: GM,
        x_features: torch.Tensor,
        training_edge_index: torch.Tensor,
        test_queries: List[str],
        ground_truth: Dict[str, Set[str]],
        use_hub: bool = True,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Perform 1-vs-all ranking evaluation for GNN model."""
        model.eval()
        n_v = self.num_papers
        all_indices = torch.arange(n_v, device=self.device)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        x_gpu = x_features.to(self.device).float()
        train_edges_gpu = training_edge_index.to(self.device)

        mrr_total = 0.0
        hits_total = 0.0
        rec_total = 0.0
        ndcg_total = 0.0
        valid_queries_count = 0

        with torch.no_grad():
            # Encoded representations: candidates connected, queries isolated
            z_connected = model.enc(x_gpu, all_indices, train_edges_gpu)
            z_isolated = model.enc(x_gpu, all_indices, empty_edge_index)

            for u in test_queries:
                if u not in self.paper_to_idx:
                    continue
                u_idx = self.paper_to_idx[u]
                gt_set = ground_truth.get(u, set())
                if not gt_set:
                    continue

                valid_queries_count += 1
                u_year = self.year_dict.get(u, 0)

                # Heuristics 1-vs-all
                h_full = self.heuristics_calculator.get_heuristics_batch(
                    src_indices=np.array([u_idx]),
                    is_full=True,
                    include_hub=use_hub,
                    device=self.device,
                ).view(n_v, 4 if use_hub else 3)

                # Score all candidates against query u
                z_u_expanded = z_isolated[u_idx].unsqueeze(0).expand(n_v, -1)
                scores = model.classifier(z_u_expanded, z_connected, h_full).squeeze()

                # Mask causal future and self
                scores[self.years_gpu > u_year] = -1e9
                scores[u_idx] = -1e9

                # Top predictions
                _, topk_indices = torch.topk(scores, k=min(100, n_v))
                ranked_papers = list(self.paper_array[topk_indices.cpu().numpy()])

                metrics = calculate_all_metrics(ranked_papers, gt_set, k=top_k)
                mrr_total += metrics["mrr"]
                hits_total += metrics[f"hits{top_k}"]
                rec_total += metrics[f"rec{top_k}"]
                ndcg_total += metrics[f"ndcg{top_k}"]

        if valid_queries_count == 0:
            return {"mrr": 0.0, f"hits{top_k}": 0.0, f"rec{top_k}": 0.0, f"ndcg{top_k}": 0.0}

        return {
            "mrr": round(mrr_total / valid_queries_count, 4),
            f"hits{top_k}": round(hits_total / valid_queries_count, 4),
            f"rec{top_k}": round(rec_total / valid_queries_count, 4),
            f"ndcg{top_k}": round(ndcg_total / valid_queries_count, 4),
        }
