import argparse
import logging
import pickle
from typing import Dict, List, Set

import numpy as np
import torch

from src.data.graph_builder import GraphSplitData
from src.evaluation.evaluator import ColdStartEvaluator
from src.features.heuristics import TopologicalHeuristicsCalculator
from src.features.negative_sampler import CausalNegativeSampler
from src.models.predictors import GM
from src.training.trainer import GNNTrainer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate GNN link prediction models.")
    parser.add_argument("--graph-pkl", type=str, default="data/processed_graph.pkl")
    parser.add_argument("--features-pt", type=str, default="data/features.pt")
    parser.add_argument(
        "--encoder",
        type=str,
        default="lightgcn",
        choices=["lightgcn", "dirgcn", "neognn", "sgc", "sage", "none"],
        help="GNN encoder backbone.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="buddy",
        choices=["buddy", "standard", "ncn"],
        help="Link prediction head.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "margin", "bpr", "infonce", "asl", "fl"],
        help="Optimization loss function.",
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        default=True,
        help="Include target in-degree hubness prior feature.",
    )
    parser.add_argument(
        "--no-hub",
        action="store_false",
        dest="hub",
        help="Exclude target in-degree hubness prior feature.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scoreboard", type=str, default="scoreboard.csv")
    args = parser.parse_args()

    # 1. Load data
    logger.info(f"Loading graph data from '{args.graph_pkl}'...")
    with open(args.graph_pkl, "rb") as f:
        graph_data: GraphSplitData = pickle.load(f)

    logger.info(f"Loading feature tensors from '{args.features_pt}'...")
    feat_payload = torch.load(args.features_pt, map_location="cpu")
    ordered_papers: List[str] = feat_payload["ordered_papers"]
    x_features: torch.Tensor = feat_payload["x_features"]

    paper_to_idx = {p: i for i, p in enumerate(ordered_papers)}
    num_nodes = len(ordered_papers)

    # 2. Extract training and validation citation edges
    train_src, train_dst = [], []
    val_src, val_dst = [], []
    train_paper_set = set(graph_data.train_papers)
    val_paper_set = set(graph_data.val_papers)
    test_paper_set = set(graph_data.test_papers)

    for u, v, d in graph_data.full_graph.edges(data=True):
        if d.get("type") == "cites" and u in paper_to_idx and v in paper_to_idx:
            ui, vi = paper_to_idx[u], paper_to_idx[v]
            if u in train_paper_set:
                train_src.append(ui)
                train_dst.append(vi)
            elif u in val_paper_set:
                val_src.append(ui)
                val_dst.append(vi)

    # Historical message passing citation edges from G_m
    m_src, m_dst = [], []
    for u, v, d in graph_data.masked_graph.edges(data=True):
        if d.get("type") == "cites" and u in paper_to_idx and v in paper_to_idx:
            m_src.append(paper_to_idx[u])
            m_dst.append(paper_to_idx[v])
    training_edge_index = torch.tensor([m_src, m_dst], dtype=torch.long)

    # Ground truth citations for evaluation
    ground_truth: Dict[str, Set[str]] = {}
    for u, v, d in graph_data.full_graph.edges(data=True):
        if d.get("type") == "cites" and u in test_paper_set:
            ground_truth.setdefault(u, set()).add(v)

    # 3. Initialize components
    heuristics_calc = TopologicalHeuristicsCalculator(
        ordered_papers=ordered_papers,
        year_dict=graph_data.year_dict,
        authors_dict=graph_data.authors_dict,
        concepts_dict=graph_data.concepts_dict,
        in_degree_dict=graph_data.train_in_degrees,
    )

    years_arr = np.array([graph_data.year_dict.get(p, 0) for p in ordered_papers])
    neg_sampler = CausalNegativeSampler(years=years_arr)

    evaluator = ColdStartEvaluator(
        ordered_papers=ordered_papers,
        year_dict=graph_data.year_dict,
        heuristics_calculator=heuristics_calc,
    )

    # 4. Instantiate Model
    num_heuristics = 4 if args.hub else 3
    model = GM(
        n_f=x_features.size(1),
        n_v=num_nodes,
        n_a=num_heuristics,
        o_d=128,
        g_type=args.encoder,
        p_type=args.predictor,
    )

    exp_name = f"{args.encoder.upper()}_{args.predictor.upper()}_{args.loss.upper()}_Hub_{args.hub}"
    trainer = GNNTrainer(
        model=model,
        heuristics_calculator=heuristics_calc,
        negative_sampler=neg_sampler,
        loss_name=args.loss,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        use_hub=args.hub,
        scoreboard_path=args.scoreboard,
    )

    results = trainer.fit_and_evaluate(
        experiment_name=exp_name,
        x_features=x_features,
        training_edges=training_edge_index,
        train_src=np.array(train_src),
        train_dst=np.array(train_dst),
        val_src=np.array(val_src),
        val_dst=np.array(val_dst),
        evaluator=evaluator,
        test_queries=graph_data.test_papers,
        ground_truth=ground_truth,
    )

    logger.info(f"Experiment finished successfully: {results}")


if __name__ == "__main__":
    main()
