import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TextAE(nn.Module):
    """Fully-connected Autoencoder for text embedding dimensionality reduction.

    Compresses concatenated SciBERT embeddings (2304d = 768 title + 768 abstract + 768 concepts)
    down to a compact 256d latent space.
    """

    def __init__(
        self,
        in_dim: int = 2304,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input embeddings."""
        return self.decoder(self.encoder(x))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Extract compressed bottleneck representation."""
        return self.encoder(x)


class SciBERTTextPipeline:
    """Manages sentence transformer encoding and autoencoder compression."""

    def __init__(
        self,
        model_name: str = "pritamdeka/S-SciBERT-snli-multinli-stsb",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._st_model: Optional[SentenceTransformer] = None

    @property
    def st_model(self) -> SentenceTransformer:
        if self._st_model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name} on {self.device}...")
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
        return self._st_model

    def encode_text_fields(
        self,
        raw_data_path: str,
        ordered_papers: List[str],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Extract and concatenate title, abstract, and concepts embeddings.

        Returns
        -------
        torch.Tensor of shape (num_papers, 2304)
        """
        logger.info(f"Reading text fields from '{raw_data_path}'...")
        metadata: Dict[str, Tuple[str, str, str]] = {}
        with open(raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                p_id = obj.get("i")
                if p_id:
                    metadata[p_id] = (
                        obj.get("ti", "") or "",
                        obj.get("ab", "") or "",
                        obj.get("ct", "") or "",
                    )

        titles = [metadata.get(p, ("", "", ""))[0] for p in ordered_papers]
        abstracts = [metadata.get(p, ("", "", ""))[1] for p in ordered_papers]
        concepts = [metadata.get(p, ("", "", ""))[2] for p in ordered_papers]

        logger.info(f"Encoding {len(ordered_papers)} titles...")
        e_title = self.st_model.encode(
            titles, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True
        )
        logger.info(f"Encoding {len(ordered_papers)} abstracts...")
        e_abstract = self.st_model.encode(
            abstracts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True
        )
        logger.info(f"Encoding {len(ordered_papers)} concepts...")
        e_concepts = self.st_model.encode(
            concepts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True
        )

        concatenated = torch.cat([e_title, e_abstract, e_concepts], dim=1).cpu()
        logger.info(f"Constructed semantic tensor: shape {concatenated.shape}")
        return concatenated

    def train_autoencoder(
        self,
        features: torch.Tensor,
        in_dim: int = 2304,
        bottleneck_dim: int = 256,
        epochs: int = 40,
        batch_size: int = 2048,
        lr: float = 1e-3,
    ) -> Tuple[TextAE, torch.Tensor]:
        """Train the TextAE model with MSE reconstruction loss and compress features."""
        model = TextAE(in_dim=in_dim, bottleneck_dim=bottleneck_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        logger.info(
            f"Training TextAE: {epochs} epochs, batch_size={batch_size}, lr={lr}..."
        )
        model.train()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

            avg_loss = total_loss / len(features)
            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} | MSE Loss: {avg_loss:.6f} | "
                    f"Elapsed: {time.time() - t0:.2f}s"
                )

        model.eval()
        logger.info("Compressing full feature set to bottleneck representation...")
        compressed_batches: List[torch.Tensor] = []
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in eval_loader:
                batch_x = batch_x.to(self.device)
                compressed = model.compress(batch_x).cpu()
                compressed_batches.append(compressed)

        compressed_features = torch.cat(compressed_batches, dim=0)
        logger.info(f"Compressed feature tensor shape: {compressed_features.shape}")
        return model, compressed_features
