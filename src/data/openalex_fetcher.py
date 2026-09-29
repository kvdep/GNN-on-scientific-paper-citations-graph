import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set
import requests

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class OpenAlexFetcher:
    """Fetcher for academic works from the OpenAlex REST API.

    Handles cursor-based pagination, inverted abstract reconstruction, concept
    filtering, rate limiting, and multithreaded downloading across publication years.
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        email: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        timeout: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        if self.email:
            self.session.headers.update({"User-Agent": f"mailto:{self.email}"})

    @staticmethod
    def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Reconstruct plain-text abstract from OpenAlex inverted index representation."""
        if not inverted_index:
            return ""
        word_positions: List[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)

    def extract_metadata(self, work: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and normalize essential metadata from an OpenAlex work record."""
        work_id = work.get("id")
        if not work_id:
            return None

        # Clean ID (e.g., "https://openalex.org/W12345" -> "W12345")
        work_id = work_id.split("/")[-1]
        year = work.get("publication_year")
        if year is None:
            return None

        title = work.get("title") or ""
        abstract = self.reconstruct_abstract(work.get("abstract_inverted_index"))

        # Concepts
        concepts = [
            c.get("display_name", "")
            for c in work.get("concepts", [])
            if c.get("display_name")
        ]
        concept_str = " ".join(concepts)

        # Authorships
        authors: List[str] = []
        for authorship in work.get("authorships", []):
            author_obj = authorship.get("author", {})
            if author_obj and author_obj.get("id"):
                authors.append(author_obj["id"].split("/")[-1])

        # Referenced works (citations)
        referenced_works: List[str] = [
            ref.split("/")[-1]
            for ref in work.get("referenced_works", [])
            if ref
        ]

        return {
            "i": work_id,
            "y": int(year),
            "ti": title.strip(),
            "ab": abstract.strip(),
            "ct": concept_str.strip(),
            "au": authors,
            "rf": referenced_works,
        }

    def fetch_year(
        self,
        year: int,
        target_count: int,
        concept_ids: Optional[List[str]] = None,
        per_page: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch papers published in a specific year using cursor pagination."""
        collected: List[Dict[str, Any]] = []
        cursor = "*"
        filters = [f"publication_year:{year}", "has_abstract:true"]

        if concept_ids:
            concept_filter = "|".join(concept_ids)
            filters.append(f"concepts.id:{concept_filter}")

        filter_param = ",".join(filters)

        while len(collected) < target_count and cursor:
            params = {
                "filter": filter_param,
                "per-page": min(per_page, target_count - len(collected)),
                "cursor": cursor,
            }

            success = False
            for attempt in range(self.max_retries):
                try:
                    time.sleep(self.rate_limit_delay)
                    response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        meta = data.get("meta", {})
                        cursor = meta.get("next_cursor")

                        for w in results:
                            item = self.extract_metadata(w)
                            if item:
                                collected.append(item)
                                if len(collected) >= target_count:
                                    break
                        success = True
                        break
                    elif response.status_code == 429:
                        wait_time = (2 ** attempt) + 1
                        logger.warning(f"Rate limited (429) for year {year}. Backing off {wait_time}s.")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status_code} for year {year}: {response.text[:200]}")
                        time.sleep(1)
                except Exception as ex:
                    logger.warning(f"Exception during request for year {year} (attempt {attempt+1}): {ex}")
                    time.sleep(2)

            if not success:
                logger.error(f"Failed to fetch batch for year {year} after {self.max_retries} retries.")
                break

        logger.info(f"Year {year}: fetched {len(collected)} records.")
        return collected

    def fetch_dataset(
        self,
        output_file: str,
        start_year: int = 2017,
        end_year: int = 2026,
        target_per_year: int = 200_000,
        concept_ids: Optional[List[str]] = None,
        max_workers: int = 4,
    ) -> None:
        """Download multithreaded dataset across years and write to JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        years = list(range(start_year, end_year + 1))
        file_lock = threading.Lock()

        logger.info(
            f"Starting dataset fetch: {len(years)} years ({start_year}-{end_year}), "
            f"target {target_per_year}/year into '{output_file}'"
        )

        with open(output_file, "w", encoding="utf-8") as out_f:
            def _worker(y: int):
                records = self.fetch_year(y, target_per_year, concept_ids=concept_ids)
                with file_lock:
                    for rec in records:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                return y, len(records)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_worker, y): y for y in years}
                for fut in as_completed(futures):
                    y = futures[fut]
                    try:
                        yr, count = fut.result()
                        logger.info(f"Completed year {yr}: {count} records written.")
                    except Exception as exc:
                        logger.error(f"Worker for year {y} generated an exception: {exc}")

        logger.info(f"Finished dataset fetch to '{output_file}'.")
