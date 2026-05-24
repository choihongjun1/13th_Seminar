"""
Training utilities for the Image-to-LaTeX formula recognition model.

Colab example
-------------
>>> from models import FormulaDenseNetEncoder, FormulaTransformerDecoder
>>> from trainer import Trainer
>>>
>>> encoder = FormulaDenseNetEncoder(hidden_dim=512)
>>> decoder = FormulaTransformerDecoder(vocab_size=len(vocab), hidden_dim=512)
>>> optimizer = torch.optim.AdamW(
...     list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
... )
>>> criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding_idx=0
>>>
>>> trainer = Trainer(
...     encoder=encoder,
...     decoder=decoder,
...     train_loader=train_loader,
...     val_loader=val_loader,
...     optimizer=optimizer,
...     criterion=criterion,
...     padding_idx=0,
...     num_epochs=20,
... )
>>> trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import jiwer
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - Colab/notebooks usually have tqdm

    class _TqdmFallback:
        """Minimal tqdm stand-in when the package is not installed."""

        def __init__(self, iterable: Iterable, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **kwargs: Any) -> None:
            pass

    def tqdm(iterable: Iterable, **kwargs: Any) -> _TqdmFallback:  # type: ignore[misc]
        return _TqdmFallback(iterable, **kwargs)

from models import FormulaDenseNetEncoder, FormulaTransformerDecoder

logger = logging.getLogger(__name__)

Batch = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Dict[str, torch.Tensor],
]

DecodeFn = Callable[[torch.Tensor], str]


class Trainer:
    """
    End-to-end trainer for ``FormulaDenseNetEncoder`` + ``FormulaTransformerDecoder``.

    Each batch must provide:
        - Images: ``[B, 1, H, W]`` grayscale tensors.
        - Tokens: ``[B, L]`` int64 token IDs (BOS … EOS, then PAD).

    Teacher forcing uses ``tokens[:, :-1]`` as decoder input and ``tokens[:, 1:]``
    as prediction targets. The decoder applies the causal mask internally; this
    class supplies ``tgt_key_padding_mask`` for padded decoder inputs.
    """

    def __init__(
        self,
        encoder: FormulaDenseNetEncoder,
        decoder: FormulaTransformerDecoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        scheduler: Optional[ReduceLROnPlateau] = None,
        padding_idx: int = 0,
        num_epochs: int = 10,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        image_key: str = "image",
        tokens_key: str = "tokens",
        grad_clip_norm: Optional[float] = 1.0,
        decode_fn: Optional[DecodeFn] = None,
        compute_metrics: bool = False,
        log_interval: int = 1,
    ) -> None:
        """
        Args:
            encoder: DenseNet image encoder.
            decoder: Transformer decoder.
            train_loader: Training ``DataLoader``.
            val_loader: Validation ``DataLoader``.
            optimizer: Optimizer over encoder + decoder parameters.
            criterion: Loss function (typically ``CrossEntropyLoss`` with
                ``ignore_index=padding_idx``).
            device: Target device; auto-detects CUDA when ``None``.
            scheduler: LR scheduler; defaults to ``ReduceLROnPlateau`` on
                validation loss when ``None``.
            padding_idx: Token ID used for padding (ignored by loss).
            num_epochs: Number of full train/val passes.
            checkpoint_dir: Directory for checkpoints.
            image_key: Dict key for images when batches are mappings.
            tokens_key: Dict key for token sequences when batches are mappings.
            grad_clip_norm: Max norm for gradient clipping; ``None`` disables.
            decode_fn: Optional ``ids -> string`` for CER/BLEU (see
                ``compute_sequence_metrics``).
            compute_metrics: If True and ``decode_fn`` is set, compute CER/BLEU
                on validation (placeholder returns ``None`` otherwise).
            log_interval: Print epoch summary every N epochs (always prints last).
        """
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.image_key = image_key
        self.tokens_key = tokens_key
        self.grad_clip_norm = grad_clip_norm
        self.decode_fn = decode_fn
        self.compute_metrics = compute_metrics
        self.log_interval = log_interval

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.scheduler = scheduler or ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_cer": [],
            "val_bleu": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Batch handling
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract images and token sequences from a loader batch.

        Supports:
            - ``(images, tokens)`` tuples/lists
            - ``{"image": ..., "tokens": ...}`` dicts (keys configurable)
        """
        if isinstance(batch, dict):
            images = batch[self.image_key]
            tokens = batch[self.tokens_key]
        elif isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError(
                    "Tuple/list batches must be (images, tokens, ...)."
                )
            images, tokens = batch[0], batch[1]
        else:
            raise TypeError(
                f"Unsupported batch type {type(batch)!r}. "
                "Use a dict or (images, tokens) tuple."
            )
        return images, tokens

    def _to_device(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.to(self.device, non_blocking=True) for t in tensors)

    @staticmethod
    def _shift_for_teacher_forcing(
        tokens: torch.Tensor,
        padding_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build decoder input, prediction targets, and padding mask.

        Args:
            tokens: Full sequences ``[B, L]`` (e.g. ``[BOS, w1, …, EOS, PAD, …]``).

        Returns:
            tgt_in:  ``tokens[:, :-1]`` — decoder input (teacher forcing).
            tgt_out: ``tokens[:, 1:]``  — logits are trained to predict these.
            tgt_key_padding_mask: ``[B, T]`` bool, True at padded input positions.
        """
        tgt_in = tokens[:, :-1]
        tgt_out = tokens[:, 1:]
        tgt_key_padding_mask = tgt_in == padding_idx
        return tgt_in, tgt_out, tgt_key_padding_mask

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _forward(
        self,
        images: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encoder → decoder forward pass.

        Args:
            images: ``[B, 1, H, W]``
            tgt_in: ``[B, T]`` shifted-right targets (teacher forcing).
            tgt_key_padding_mask: ``[B, T]`` bool padding mask.

        Returns:
            Logits ``[B, T, vocab_size]``. Causal mask is applied inside the decoder.
        """
        memory = self.encoder(images)
        logits = self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    def _compute_loss(
        self,
        logits: torch.Tensor,
        tgt_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy over vocabulary; padding tokens in ``tgt_out`` are ignored
        when ``criterion.ignore_index == padding_idx``.

        Args:
            logits: ``[B, T, vocab_size]``
            tgt_out: ``[B, T]``

        Returns:
            Scalar loss.
        """
        # CrossEntropyLoss expects [B, C, T] and targets [B, T].
        return self.criterion(logits.transpose(1, 2), tgt_out)

    # ------------------------------------------------------------------
    # Metrics (placeholders + optional decode)
    # ------------------------------------------------------------------

    def compute_sequence_metrics(
        self,
        pred_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[str, Optional[float]]:
        """
        Compute batch-averaged CER and BLEU from predicted and target token IDs.

        Each row in ``pred_ids`` / ``target_ids`` is decoded with ``decode_fn``
        and scored with ``jiwer.cer`` and ``nltk`` sentence BLEU (with smoothing).

        Returns:
            Dict with keys ``cer`` and ``bleu`` (``None`` if ``decode_fn`` is unset
            or the batch is empty).
        """
        if self.decode_fn is None:
            return {"cer": None, "bleu": None}

        batch_size = pred_ids.size(0)
        if batch_size == 0:
            return {"cer": None, "bleu": None}

        smoothing = SmoothingFunction().method1
        cer_scores: list[float] = []
        bleu_scores: list[float] = []

        for i in range(batch_size):
            pred_str = self.decode_fn(pred_ids[i]) or ""
            target_str = self.decode_fn(target_ids[i]) or ""

            if not target_str and not pred_str:
                cer_scores.append(0.0)
            elif not target_str or not pred_str:
                cer_scores.append(1.0)
            else:
                cer_scores.append(float(jiwer.cer(target_str, pred_str)))

            ref_tokens = target_str.split()
            hyp_tokens = pred_str.split()

            if not ref_tokens and not hyp_tokens:
                bleu_scores.append(1.0)
            elif not ref_tokens or not hyp_tokens:
                bleu_scores.append(0.0)
            else:
                bleu_scores.append(
                    float(
                        sentence_bleu(
                            [ref_tokens],
                            hyp_tokens,
                            smoothing_function=smoothing,
                        )
                    )
                )

        return {
            "cer": sum(cer_scores) / len(cer_scores),
            "bleu": sum(bleu_scores) / len(bleu_scores),
        }

    @torch.no_grad()
    def _greedy_decode_ids(self, logits: torch.Tensor) -> torch.Tensor:
        """Argmax token IDs from logits ``[B, T, vocab_size]``."""
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch; return mean loss."""
        self.encoder.train()
        self.decoder.train()

        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train {epoch + 1}/{self.num_epochs}",
            leave=False,
        )

        for batch in pbar:
            images, tokens = self._unpack_batch(batch)
            images, tokens = self._to_device(images, tokens)

            tgt_in, tgt_out, tgt_pad_mask = self._shift_for_teacher_forcing(
                tokens, self.padding_idx
            )

            self.optimizer.zero_grad(set_to_none=True)

            logits = self._forward(images, tgt_in, tgt_pad_mask)
            loss = self._compute_loss(logits, tgt_out)

            loss.backward()

            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.grad_clip_norm,
                )

            self.optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, Optional[float]]]:
        """Run validation; return mean loss and optional metrics."""
        self.encoder.eval()
        self.decoder.eval()

        running_loss = 0.0
        num_batches = 0
        metric_accum: Dict[str, list] = {"cer": [], "bleu": []}

        pbar = tqdm(
            self.val_loader,
            desc=f"Val   {epoch + 1}/{self.num_epochs}",
            leave=False,
        )

        for batch in pbar:
            images, tokens = self._unpack_batch(batch)
            images, tokens = self._to_device(images, tokens)

            tgt_in, tgt_out, tgt_pad_mask = self._shift_for_teacher_forcing(
                tokens, self.padding_idx
            )

            logits = self._forward(images, tgt_in, tgt_pad_mask)
            loss = self._compute_loss(logits, tgt_out)

            batch_loss = loss.item()
            running_loss += batch_loss
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

            if self.compute_metrics and self.decode_fn is not None:
                pred_ids = self._greedy_decode_ids(logits)
                batch_metrics = self.compute_sequence_metrics(pred_ids, tgt_out)
                for key in ("cer", "bleu"):
                    if batch_metrics.get(key) is not None:
                        metric_accum[key].append(batch_metrics[key])

        avg_loss = running_loss / max(num_batches, 1)
        avg_metrics = {
            "cer": (
                sum(metric_accum["cer"]) / len(metric_accum["cer"])
                if metric_accum["cer"]
                else None
            ),
            "bleu": (
                sum(metric_accum["bleu"]) / len(metric_accum["bleu"])
                if metric_accum["bleu"]
                else None
            ),
        }
        return avg_loss, avg_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> Path:
        """Save encoder/decoder weights, optimizer, and training state."""
        state = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "padding_idx": self.padding_idx,
        }

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info("Saved new best checkpoint to %s", best_path)
            return best_path

        return latest_path

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
    ) -> int:
        """
        Restore weights (and optionally optimizer/scheduler) from a checkpoint.

        Returns:
            The epoch index stored in the checkpoint.
        """
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return int(checkpoint.get("epoch", 0))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self, start_epoch: int = 0) -> Dict[str, list]:
        """
        Run the full training schedule (train + val each epoch).

        Returns:
            History dict with per-epoch ``train_loss``, ``val_loss``, ``lr``,
            and optional ``val_cer`` / ``val_bleu``.
        """
        logger.info("Training on device: %s", self.device)

        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_cer"].append(val_metrics["cer"])
            self.history["val_bleu"].append(val_metrics["bleu"])
            self.history["lr"].append(current_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch, val_loss, is_best=is_best)

            if (epoch + 1) % self.log_interval == 0 or (epoch + 1) == self.num_epochs:
                cer_str = (
                    f"{val_metrics['cer']:.4f}"
                    if val_metrics["cer"] is not None
                    else "n/a"
                )
                bleu_str = (
                    f"{val_metrics['bleu']:.4f}"
                    if val_metrics["bleu"] is not None
                    else "n/a"
                )
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}] "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"lr={current_lr:.2e}  CER={cer_str}  BLEU={bleu_str}"
                    + ("  *best*" if is_best else "")
                )

        print(
            f"Training complete. Best val_loss={self.best_val_loss:.4f} "
            f"→ {self.checkpoint_dir / 'best.pt'}"
        )
        return self.history


def build_criterion(
    padding_idx: int = 0,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """
    Factory for the standard training loss (ignores padding tokens).

    Args:
        padding_idx: Index to ignore in targets.
        label_smoothing: Optional label smoothing (PyTorch >= 1.10).
    """
    return nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=label_smoothing)
