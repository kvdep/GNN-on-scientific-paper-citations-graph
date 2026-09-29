import argparse
import logging
from src.data.openalex_fetcher import OpenAlexFetcher

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch academic papers from OpenAlex API.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_data.jsonl",
        help="Target output path for JSONL records.",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017, help="Starting publication year."
    )
    parser.add_argument(
        "--end-year", type=int, default=2026, help="Ending publication year."
    )
    parser.add_argument(
        "--target-per-year",
        type=int,
        default=200_000,
        help="Number of records to fetch per year.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent download threads."
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Polite pool email for OpenAlex rate-limit boost.",
    )
    args = parser.parse_args()

    # Default OpenAlex concept IDs for Machine Learning / Computer Science
    ai_concepts = ["C154945302", "C41008148"]

    fetcher = OpenAlexFetcher(email=args.email)
    fetcher.fetch_dataset(
        output_file=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        target_per_year=args.target_per_year,
        concept_ids=ai_concepts,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
