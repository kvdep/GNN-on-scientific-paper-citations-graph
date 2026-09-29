import argparse
import logging
import pickle
from typing import Dict, List, Set

import numpy as np
import torch

from src.data.graph_builder import GraphSplitData
from src.evaluation.metrics import calculate_all_metrics
from src.features.negative_sampler import CausalNegativeSampler
from src.models.gbdt_ranker import CatBoostCitationRanker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CatBoost citation ranker.")
    parser.add_argument("--graph-pkl", type=str, default="data/processed_graph.pkl")
    parser.add_argument("--features-pt", type=str, default="data/features.pt")
    parser.add_argument(
        "--loss-function",
        type=str,
        default="YetiRank",
        choices=["YetiRank", "QueryRMSE"],
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--candidate-pool", type=int, default=100)
    args = parser.parse_args()

    # 1. Load data
    with open(args.graph_pkl, "rb") as f:
        graph_data: GraphSplitData = pickle.load(f)

    feat_payload = torch.load(args.features_pt, map_location="cpu")
    ordered_papers: List[str] = feat_payload["ordered_papers"]
    text_embs: torch.Tensor = feat_payload["text_compressed"]
    author_topo: torch.Tensor = feat_payload["author_topo"]

    paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}
    train_paper_set = set(graph_data.train_papers)
    val_paper_set = set(graph_data.val_papers)
    test_paper_set = set(graph_data.test_papers)

    # 2. Collect edges
    train_u, train_v = [], []
    val_u, val_v = [], []
    for u, v, d in graph_data.full_graph.edges(data=True):
        if d.get("type") == "cites" and u in paper_to_idx and v in paper_to_idx:
            ui, vi = paper_to_idx[u], paper_to_idx[v]
            if u in train_paper_set:
                train_u.append(ui)
                train_v.append(vi)
            elif u in val_paper_set:
                val_u.append(ui)
                val_v.append(vi)

    years_arr = np.array([graph_data.year_dict.get(p, 0) for p in ordered_papers])
    sampler = CausalNegativeSampler(years=years_arr)

    # Sample negatives
    train_neg_v = sampler.sample_causal_negatives(np.array(train_u))
    val_neg_v = sampler.sample_causal_negatives(np.array(val_u))

    # Build pairs
    all_train_q = np.concatenate([train_u, train_u])
    all_train_cand = np.concatenate([train_v, train_neg_v])
    all_train_y = np.concatenate([np.ones(len(train_u)), np.zeros(len(train_u))])

    all_val_q = np.concatenate([val_u, val_u])
    all_val_cand = np.concatenate([val_v, val_neg_v])
    all_val_y = np.concatenate([np.ones(len(val_u)), np.zeros(len(val_u))])

    # 3. Extract 390 features
    logger.info("Extracting 390-dimensional features for training pool...")
    X_train = CatBoostCitationRanker.extract_pair_features(
        query_indices=all_train_q,
        candidate_indices=all_train_cand,
        text_embeddings=text_embs,
        author_topology_embeddings=author_topo,
        year_dict=graph_data.year_dict,
        authors_dict=graph_data.authors_dict,
        concepts_dict=graph_data.concepts_dict,
        in_degree_dict=graph_data.train_in_degrees,
        ordered_papers=ordered_papers,
    )

    logger.info("Extracting 390-dimensional features for validation pool...")
    X_val = CatBoostCitationRanker.extract_pair_features(
        query_indices=all_val_q,
        candidate_indices=all_val_cand,
        text_embeddings=text_embs,
        author_topology_embeddings=author_topo,
        year_dict=graph_data.year_dict,
        authors_dict=graph_data.authors_dict,
        concepts_dict=graph_data.concepts_dict,
        in_degree_dict=graph_data.train_in_degrees,
        ordered_papers=ordered_papers,
    )

    # 4. Train CatBoost
    ranker = CatBoostCitationRanker(
        loss_function=args.loss_function,
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
    )
    ranker.fit(
        X_train=X_train,
        y_train=all_train_y,
        queries_train=all_train_q,
        X_val=X_val,
        y_val=all_val_y,
        queries_val=all_val_q,
    )

    # 5. Evaluate on Cold Start test queries
    logger.info(f"Evaluating CatBoost ranker across {len(test_paper_set)} test papers...")
    ground_truth: Dict[str, Set[str]] = {}
    for u, v, d in graph_data.full_graph.edges(data=True):
        if d.get("type") == "cites" and u in test_paper_set:
            ground_truth.setdefault(u, set()).add(v)

    norm_text = torch.nn.functional.normalize(text_embs, p=2, dim=1)
    years_t = torch.tensor(years_arr)

    mrr_total = 0.0
    hits_total = 0.0
    rec_total = 0.0
    ndcg_total = 0.0
    count = 0

    for u in graph_data.test_papers:
        if u not in paper_to_idx or u not in ground_truth:
            continue
        u_idx = paper_to_idx[u]
        gt_set = ground_truth[u]

        # Stage 1: Semantic retrieval
        sims = torch.mm(norm_text[u_idx].unsqueeze(0), norm_text.t()).squeeze(0)
        sims[years_t > years_arr[u_idx]] = -2.0
        sims[u_idx] = -2.0

        _, cands_idx = torch.topk(sims, min(args.candidate_pool, len(ordered_papers)))
        cands_cpu = cands_idx.numpy()

        q_arr = np.full(len(cands_cpu), u_idx, dtype=np.int64)
        X_test = CatBoostCitationRanker.extract_pair_features(
            query_indices=q_arr,
            candidate_indices=cands_cpu,
            text_embeddings=text_embs,
            author_topology_embeddings=author_topo,
            year_dict=graph_data.year_dict,
            authors_dict=graph_data.authors_dict,
            concepts_dict=graph_data.concepts_dict,
            in_degree_dict=graph_data.train_in_degrees,
            ordered_papers=ordered_papers,
        )

        scores = ranker.predict(X_test)
        top_ranked_indices = cands_cpu[np.argsort(-scores)]
        ranked_papers = [ordered_papers[idx] for idx in top_ranked_indices]

        metrics = calculate_all_metrics(ranked_papers, gt_set, k=10)
        mrr_total += metrics["mrr"]
        hits_total += metrics["hits10"]
        rec_total += metrics["rec10"]
        ndcg_total += metrics["ndcg10"]
        count += 1

    if count > 0:
        logger.info(
            f"CatBoost ({args.loss_function}) Results: "
            f"MRR={mrr_total/count:.4f}, Hits@10={hits_total/count:.4f}, "
            f"Rec@10={rec_total/count:.4f}, NDCG@10={ndcg_total/count:.4f}"
        )


if __name__ == "__main__":
    main()
