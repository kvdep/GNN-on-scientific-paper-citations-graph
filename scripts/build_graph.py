import argparse
import logging
import os
import pickle
from src.data.graph_builder import CitationGraphBuilder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and partition citation graph from JSONL.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_data.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_graph.pkl",
        help="Output path for pickled GraphSplitData.",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=3,
        help="k-core pruning parameter for bipartite author-paper graph.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.85, help="Train set ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation set ratio."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10, help="Test set ratio."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random split seed."
    )
    args = parser.parse_args()

    builder = CitationGraphBuilder(
        k_core=args.k_core,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    data = builder.build_and_split(args.input)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Successfully saved GraphSplitData to '{args.output}'.")


if __name__ == "__main__":
    main()
