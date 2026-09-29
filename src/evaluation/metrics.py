import math
from typing import Dict, List, Set, Union


def mean_reciprocal_rank(positive_positions_1based: List[int]) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    MRR is the dataset average of 1 / rank of the first relevant candidate.
    """
    if not positive_positions_1based:
        return 0.0
    first_rank = min(positive_positions_1based)
    return 1.0 / first_rank


def hits_at_k(positive_positions_1based: List[int], k: int = 10) -> float:
    """Compute Hits@K for a single query.

    Returns 1.0 if at least one relevant document appears in top-K, 0.0 otherwise.
    """
    if not positive_positions_1based:
        return 0.0
    return 1.0 if any(p <= k for p in positive_positions_1based) else 0.0


def recall_at_k(num_hits_in_top_k: int, total_positives: int) -> float:
    """Compute Recall@K for a single query."""
    if total_positives == 0:
        return 0.0
    return num_hits_in_top_k / total_positives


def ndcg_at_k(
    positive_positions_1based: List[int], total_positives: int, k: int = 10
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    DCG@K = sum_{p in pos, p <= k} 1 / log2(p + 1)
    IDCG@K = sum_{p=1}^{min(|GT|, k)} 1 / log2(p + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if total_positives == 0 or not positive_positions_1based:
        return 0.0

    dcg = sum(1.0 / math.log2(p + 1) for p in positive_positions_1based if p <= k)
    idcg_limit = min(total_positives, k)
    idcg = sum(1.0 / math.log2(p + 1) for p in range(1, idcg_limit + 1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def calculate_all_metrics(
    ranked_predictions: List[str],
    ground_truth_set: Set[str],
    k: int = 10,
) -> Dict[str, float]:
    """Calculate MRR, Hits@K, Recall@K, and NDCG@K for a single query ranking."""
    if not ground_truth_set:
        return {"mrr": 0.0, f"hits{k}": 0.0, f"rec{k}": 0.0, f"ndcg{k}": 0.0}

    # Find 1-based positions of all ground truth items in predictions
    pos_1based = [
        idx + 1
        for idx, item in enumerate(ranked_predictions)
        if item in ground_truth_set
    ]

    top_k_items = set(ranked_predictions[:k])
    num_hits = len(top_k_items & ground_truth_set)
    total_pos = len(ground_truth_set)

    mrr = mean_reciprocal_rank(pos_1based)
    hits = hits_at_k(pos_1based, k=k)
    rec = recall_at_k(num_hits, total_pos)
    ndcg = ndcg_at_k(pos_1based, total_pos, k=k)

    return {
        "mrr": mrr,
        f"hits{k}": hits,
        f"rec{k}": rec,
        f"ndcg{k}": ndcg,
    }
