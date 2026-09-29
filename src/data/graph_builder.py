import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GraphSplitData:
    """Encapsulates partitioned graph structures and associated metadata."""

    full_graph: nx.DiGraph
    masked_graph: nx.DiGraph
    train_papers: List[str]
    val_papers: List[str]
    test_papers: List[str]
    year_dict: Dict[str, int]
    concepts_dict: Dict[str, Set[str]]
    authors_dict: Dict[str, Set[str]]
    train_in_degrees: Dict[str, int]


class CitationGraphBuilder:
    """Constructs heterogeneous citation and co-authorship graphs.

    Performs k-core filtering, temporal validation, randomized masking split
    (train/val/test), and calculates prior hubness metrics.
    """

    def __init__(
        self,
        k_core: int = 3,
        train_ratio: float = 0.85,
        val_ratio: float = 0.05,
        test_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        self.k_core = k_core
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def build_and_split(self, data_path: str) -> GraphSplitData:
        """Parse raw JSONL, apply k-core filtering, and construct split graphs.

        Parameters
        ----------
        data_path : str
            Path to the cleaned OpenAlex JSONL file.

        Returns
        -------
        GraphSplitData
            Object containing full graph, message-passing masked graph, splits,
            and feature dictionaries.
        """
        logger.info(f"Loading raw records from '{data_path}'...")
        paper_ids: List[str] = []
        year_dict: Dict[str, int] = {}
        concepts_dict: Dict[str, Set[str]] = {}
        authors_dict: Dict[str, Set[str]] = {}
        raw_citations: List[Tuple[str, str]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                p_id = record.get("i")
                if not p_id:
                    continue

                paper_ids.append(p_id)
                year_dict[p_id] = record.get("y", 0)

                ct_raw = record.get("ct", "")
                concepts_dict[p_id] = (
                    set(ct_raw.lower().split()) if ct_raw else set()
                )

                au_raw = record.get("au", [])
                authors_dict[p_id] = set(a for a in au_raw if a is not None)

                for ref in record.get("rf", []):
                    if ref is not None:
                        raw_citations.append((p_id, ref))

        logger.info(
            f"Read {len(paper_ids)} papers, {len(raw_citations)} raw citations."
        )

        # 1. Construct bipartite author-paper graph for k-core pruning
        logger.info(f"Building bipartite author-paper graph for k-core (k={self.k_core})...")
        bipartite_graph = nx.Graph()
        for p in paper_ids:
            for a in authors_dict[p]:
                bipartite_graph.add_edge(p, a)

        core_graph = nx.k_core(bipartite_graph, k=self.k_core)
        valid_nodes = set(core_graph.nodes())

        # Filter papers and authors retained in k-core
        filtered_papers = [p for p in paper_ids if p in valid_nodes]
        paper_set = set(filtered_papers)
        year_dict = {p: year_dict[p] for p in filtered_papers}
        concepts_dict = {p: concepts_dict[p] for p in filtered_papers}
        authors_dict = {
            p: set(a for a in authors_dict[p] if a in valid_nodes)
            for p in filtered_papers
        }
        filtered_citations = [
            (u, v) for u, v in raw_citations if u in paper_set and v in paper_set
        ]

        logger.info(
            f"After k-core (k={self.k_core}): {len(filtered_papers)} papers, "
            f"{len(filtered_citations)} valid citation edges."
        )

        # 2. Build full directed heterogeneous graph
        full_graph = nx.DiGraph()
        for p in filtered_papers:
            full_graph.add_node(p, type="paper", y=year_dict[p])
            for a in authors_dict[p]:
                full_graph.add_node(a, type="author")
                full_graph.add_edge(a, p, type="writes")
                full_graph.add_edge(p, a, type="writes")

        for u, v in filtered_citations:
            full_graph.add_edge(u, v, type="cites")

        full_graph.remove_edges_from(list(nx.selfloop_edges(full_graph)))

        # 3. Partitioning: Randomized Masked Split (Train / Val / Test)
        random.seed(self.random_seed)
        shuffled_papers = list(filtered_papers)
        random.shuffle(shuffled_papers)

        num_total = len(shuffled_papers)
        num_test = int(num_total * self.test_ratio)
        num_val = int(num_total * self.val_ratio)

        test_papers = shuffled_papers[:num_test]
        val_papers = shuffled_papers[num_test : num_test + num_val]
        train_papers = shuffled_papers[num_test + num_val :]

        test_set = set(test_papers)
        val_set = set(val_papers)
        train_set = set(train_papers)

        logger.info(
            f"Split sizes: Train={len(train_papers)} ({(len(train_papers)/num_total):.1%}), "
            f"Val={len(val_papers)} ({(len(val_papers)/num_total):.1%}), "
            f"Test={len(test_papers)} ({(len(test_papers)/num_total):.1%})"
        )

        # 4. Construct message-passing graph G_m by masking val and test citation edges
        masked_graph = full_graph.copy()
        edges_to_remove = []
        for u, v, d in masked_graph.edges(data=True):
            if d.get("type") == "cites":
                if u in test_set or u in val_set or v in test_set or v in val_set:
                    edges_to_remove.append((u, v))
        masked_graph.remove_edges_from(edges_to_remove)

        # 5. Compute in-degree hubness prior strictly from training citations
        train_in_degrees = {p: 0 for p in filtered_papers}
        for u, v, d in masked_graph.edges(data=True):
            if d.get("type") == "cites" and v in train_in_degrees:
                train_in_degrees[v] += 1

        train_cites_count = sum(
            1 for _, _, d in masked_graph.edges(data=True) if d.get("type") == "cites"
        )
        logger.info(
            f"Masked graph G_m: {masked_graph.number_of_nodes()} total nodes, "
            f"{train_cites_count} active training citation edges."
        )

        return GraphSplitData(
            full_graph=full_graph,
            masked_graph=masked_graph,
            train_papers=train_papers,
            val_papers=val_papers,
            test_papers=test_papers,
            year_dict=year_dict,
            concepts_dict=concepts_dict,
            authors_dict=authors_dict,
            train_in_degrees=train_in_degrees,
        )
