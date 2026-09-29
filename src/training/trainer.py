import csv
import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.evaluation.evaluator import ColdStartEvaluator
from src.features.heuristics import TopologicalHeuristicsCalculator
from src.features.negative_sampler import CausalNegativeSampler
from src.models.losses import ASL, BPRLoss, FL, InfoNCELoss, MarginLoss
from src.models.predictors import GM

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Orchestrates inductive GNN training and experiment checkpointing."""

    def __init__(
        self,
        model: GM,
        heuristics_calculator: TopologicalHeuristicsCalculator,
        negative_sampler: CausalNegativeSampler,
        loss_name: str = "bce",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        max_epochs: int = 20,
        patience: int = 3,
        use_hub: bool = True,
        output_dir: str = "./checkpoints",
        scoreboard_path: Optional[str] = "./scoreboard.csv",
        device: str = "cuda",
    ) -> None:
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.model = model.to(self.device)
        self.heuristics_calculator = heuristics_calculator
        self.negative_sampler = negative_sampler
        self.loss_name = loss_name.lower()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_hub = use_hub
        self.output_dir = output_dir
        self.scoreboard_path = scoreboard_path
        os.makedirs(self.output_dir, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = self._init_loss(self.loss_name)

    def _init_loss(self, loss_name: str) -> nn.Module:
        if loss_name == "margin":
            return MarginLoss(margin=0.2)
        elif loss_name == "bpr":
            return BPRLoss()
        elif loss_name == "infonce":
            return InfoNCELoss(tau=0.1)
        elif loss_name == "asl":
            return ASL(gamma_pos=1.0, gamma_neg=2.0)
        elif loss_name == "fl":
            return FL(alpha=1.0, gamma=2.0)
        return nn.BCELoss()

    def train_epoch(
        self,
        x_features: torch.Tensor,
        training_edges: torch.Tensor,
        train_src: np.ndarray,
        train_dst: np.ndarray,
    ) -> float:
        """Run single training epoch under cold-start isolated/connected paradigm."""
        self.model.train()
        n_v = x_features.size(0)
        all_indices = torch.arange(n_v, device=self.device)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        train_edges_gpu = training_edges.to(self.device)
        x_gpu = x_features.to(self.device).float()

        # Sample causal negatives
        neg_dst = self.negative_sampler.sample_causal_negatives(train_src)

        perm = np.random.permutation(len(train_src))
        epoch_loss = 0.0

        for i in range(0, len(train_src), self.batch_size):
            batch_idx = perm[i : i + self.batch_size]
            b_u = train_src[batch_idx]
            b_v_pos = train_dst[batch_idx]
            b_v_neg = neg_dst[batch_idx]

            self.optimizer.zero_grad()

            # Connected graph embeddings for candidates, isolated for queries
            z_connected = self.model.enc(x_gpu, all_indices, train_edges_gpu)
            z_isolated = self.model.enc(x_gpu, all_indices, empty_edge_index)

            h_pos = self.heuristics_calculator.get_heuristics_batch(
                b_u, b_v_pos, is_full=False, include_hub=self.use_hub, device=self.device
            )
            h_neg = self.heuristics_calculator.get_heuristics_batch(
                b_u, b_v_neg, is_full=False, include_hub=self.use_hub, device=self.device
            )

            pos_scores = self.model.classifier(
                z_isolated[b_u], z_connected[b_v_pos], h_pos
            ).squeeze()
            neg_scores = self.model.classifier(
                z_isolated[b_u], z_connected[b_v_neg], h_neg
            ).squeeze()

            if isinstance(self.criterion, (MarginLoss, BPRLoss, InfoNCELoss)):
                combined_scores = torch.cat([pos_scores, neg_scores])
                combined_targets = torch.cat(
                    [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
                )
                loss = self.criterion(combined_scores, combined_targets)
            else:
                loss = self.criterion(
                    pos_scores, torch.ones_like(pos_scores)
                ) + self.criterion(neg_scores, torch.zeros_like(neg_scores))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item() * len(b_u)

        return epoch_loss / len(train_src)

    def validate_epoch(
        self,
        x_features: torch.Tensor,
        training_edges: torch.Tensor,
        val_src: np.ndarray,
        val_dst: np.ndarray,
    ) -> float:
        """Run validation loss calculation."""
        self.model.eval()
        n_v = x_features.size(0)
        all_indices = torch.arange(n_v, device=self.device)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        train_edges_gpu = training_edges.to(self.device)
        x_gpu = x_features.to(self.device).float()

        neg_dst_val = self.negative_sampler.sample_causal_negatives(val_src)

        with torch.no_grad():
            z_connected = self.model.enc(x_gpu, all_indices, train_edges_gpu)
            z_isolated = self.model.enc(x_gpu, all_indices, empty_edge_index)

            h_pos = self.heuristics_calculator.get_heuristics_batch(
                val_src, val_dst, is_full=False, include_hub=self.use_hub, device=self.device
            )
            h_neg = self.heuristics_calculator.get_heuristics_batch(
                val_src, neg_dst_val, is_full=False, include_hub=self.use_hub, device=self.device
            )

            pos_scores = self.model.classifier(
                z_isolated[val_src], z_connected[val_dst], h_pos
            ).squeeze()
            neg_scores = self.model.classifier(
                z_isolated[val_src], z_connected[neg_dst_val], h_neg
            ).squeeze()

            if isinstance(self.criterion, (MarginLoss, BPRLoss, InfoNCELoss)):
                combined_scores = torch.cat([pos_scores, neg_scores])
                combined_targets = torch.cat(
                    [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
                )
                val_loss = self.criterion(combined_scores, combined_targets).item()
            else:
                val_loss = (
                    self.criterion(pos_scores, torch.ones_like(pos_scores))
                    + self.criterion(neg_scores, torch.zeros_like(neg_scores))
                ).item()

        return val_loss

    def fit_and_evaluate(
        self,
        experiment_name: str,
        x_features: torch.Tensor,
        training_edges: torch.Tensor,
        train_src: np.ndarray,
        train_dst: np.ndarray,
        val_src: np.ndarray,
        val_dst: np.ndarray,
        evaluator: ColdStartEvaluator,
        test_queries: List[str],
        ground_truth: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        """Complete training routine with early stopping and evaluation."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.output_dir, f"{experiment_name}_best.pt")

        logger.info(f"Starting training for {experiment_name} ({self.max_epochs} epochs)...")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr_loss = self.train_epoch(x_features, training_edges, train_src, train_dst)
            vl_loss = self.validate_epoch(x_features, training_edges, val_src, val_dst)
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | "
                f"Val Loss: {vl_loss:.4f} | Time: {elapsed:.2f}s"
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best checkpoint and evaluate
        logger.info(f"Evaluating best checkpoint from '{best_model_path}'...")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        metrics = evaluator.evaluate_gnn(
            model=self.model,
            x_features=x_features,
            training_edge_index=training_edges,
            test_queries=test_queries,
            ground_truth=ground_truth,
            use_hub=self.use_hub,
        )

        logger.info(f"Final Test Metrics for {experiment_name}: {metrics}")

        # Update scoreboard
        if self.scoreboard_path:
            self._update_scoreboard(experiment_name, metrics)

        return {"experiment": experiment_name, "metrics": metrics, "best_loss": best_val_loss}

    def _update_scoreboard(self, experiment_name: str, metrics: Dict[str, float]) -> None:
        """Append test evaluation results to CSV scoreboard."""
        file_exists = os.path.exists(self.scoreboard_path)
        row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "experiment": experiment_name,
            **metrics,
        }
        fieldnames = ["date", "experiment", "mrr", "hits10", "rec10", "ndcg10"]
        with open(self.scoreboard_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Scoreboard updated at '{self.scoreboard_path}'.")
