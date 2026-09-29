import argparse
import logging
import os
import pickle
import torch

from src.data.graph_builder import GraphSplitData
from src.features.fastrp import FastRP
from src.features.text_encoder import SciBERTTextPipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SciBERT, AutoEncoder, and FastRP features."
    )
    parser.add_argument(
        "--graph-pkl",
        type=str,
        default="data/processed_graph.pkl",
        help="Path to processed_graph.pkl.",
    )
    parser.add_argument(
        "--raw-jsonl",
        type=str,
        default="data/raw_data.jsonl",
        help="Path to raw JSONL data.",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="data/features.pt",
        help="Target output path for PyTorch features.",
    )
    parser.add_argument(
        "--fastrp-dim", type=int, default=128, help="FastRP embedding dimension."
    )
    parser.add_argument(
        "--ae-dim", type=int, default=256, help="TextAE bottleneck dimension."
    )
    parser.add_argument(
        "--ae-epochs", type=int, default=40, help="Number of TextAE training epochs."
    )
    args = parser.parse_args()

    logger.info(f"Loading GraphSplitData from '{args.graph_pkl}'...")
    with open(args.graph_pkl, "rb") as f:
        graph_data: GraphSplitData = pickle.load(f)

    # Order all papers consistently
    ordered_papers = (
        graph_data.train_papers + graph_data.val_papers + graph_data.test_papers
    )

    # 1. Text Pipeline: SciBERT extraction + AutoEncoder compression
    text_pipe = SciBERTTextPipeline()
    raw_text_embs = text_pipe.encode_text_fields(args.raw_jsonl, ordered_papers)
    ae_model, compressed_text_embs = text_pipe.train_autoencoder(
        features=raw_text_embs,
        in_dim=2304,
        bottleneck_dim=args.ae_dim,
        epochs=args.ae_epochs,
    )

    # 2. Topology Pipeline: FastRP (128d)
    fastrp = FastRP(dim=args.fastrp_dim)
    topo_embs = fastrp.compute(graph_data.masked_graph, ordered_papers)

    # 3. Author Topology Aggregation (128d)
    author_topo_embs = FastRP.aggregate_author_topology(
        ordered_papers, graph_data.authors_dict, topo_embs
    )

    # 4. Joint Feature Matrix: Text (256d) + FastRP (128d) = 384d
    joint_features = torch.cat([compressed_text_embs, topo_embs], dim=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)), exist_ok=True)
    payload = {
        "ordered_papers": ordered_papers,
        "x_features": joint_features,
        "text_compressed": compressed_text_embs,
        "topo_fastrp": topo_embs,
        "author_topo": author_topo_embs,
    }
    torch.save(payload, args.output_pt)
    logger.info(f"Successfully saved feature tensors to '{args.output_pt}'.")


if __name__ == "__main__":
    main()
