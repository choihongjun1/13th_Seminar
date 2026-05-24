"""
Autoregressive evaluation on the IM2LATEX test split.

Loads ``best.pt``, runs ``decoder.generate()`` over ``test_loader``, and reports
CER, BLEU, and exact-match accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Union

import jiwer
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from im2latex_data_pipeline import (
    LatexDataset,
    Vocab,
    build_vocab_from_train,
    create_dataloader,
    gray_normalize_neg1_pos1,
)
from models import FormulaDenseNetEncoder, FormulaTransformerDecoder

DecodeFn = Callable[[torch.Tensor], str]


def make_decode_fn(vocab: Vocab) -> DecodeFn:
    """Build a single-sequence decode function from a ``Vocab``."""

    def decode_fn(ids: torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        clean_ids: list[int] = []
        for idx in ids:
            if idx == vocab.eos_id:
                break
            clean_ids.append(idx)
        return vocab.decode(clean_ids, skip_special=True)

    return decode_fn


def _score_strings(pred_str: str, target_str: str) -> tuple[float, float, float]:
    """Return (cer, bleu, exact_match) for one pair of decoded strings."""
    pred_str = pred_str or ""
    target_str = target_str or ""

    if not target_str and not pred_str:
        cer = 0.0
    elif not target_str or not pred_str:
        cer = 1.0
    else:
        cer = float(jiwer.cer(target_str, pred_str))

    ref_tokens = target_str.split()
    hyp_tokens = pred_str.split()
    smoothing = SmoothingFunction().method1

    if not ref_tokens and not hyp_tokens:
        bleu = 1.0
    elif not ref_tokens or not hyp_tokens:
        bleu = 0.0
    else:
        bleu = float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=smoothing,
            )
        )

    exact = float(pred_str == target_str)
    return cer, bleu, exact


@torch.no_grad()
def evaluate_test_set(
    encoder: FormulaDenseNetEncoder,
    decoder: FormulaTransformerDecoder,
    test_loader: DataLoader,
    decode_fn: DecodeFn,
    sos_id: int,
    eos_id: int,
    device: torch.device,
    max_gen_len: int = 256,
) -> Dict[str, float]:
    """
    Run greedy autoregressive inference and aggregate test metrics.

    Expects batches from ``collate_formulas`` with keys ``images`` and
    ``formula_ids``.
    """
    encoder.eval()
    decoder.eval()

    cer_sum = 0.0
    bleu_sum = 0.0
    exact_sum = 0.0
    num_samples = 0

    for batch in tqdm(test_loader, desc="Test eval"):
        images = batch["images"].to(device, non_blocking=True)
        target_ids = batch["formula_ids"].to(device, non_blocking=True)

        memory = encoder(images)
        pred_ids = decoder.generate(
            memory,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_gen_len,
        )

        batch_size = images.size(0)
        for i in range(batch_size):
            pred_str = decode_fn(pred_ids[i]) or ""
            target_str = decode_fn(target_ids[i]) or ""
            cer, bleu, exact = _score_strings(pred_str, target_str)
            cer_sum += cer
            bleu_sum += bleu
            exact_sum += exact
            num_samples += 1

    if num_samples == 0:
        return {"cer": 0.0, "bleu": 0.0, "exact_match": 0.0}

    return {
        "cer": cer_sum / num_samples,
        "bleu": bleu_sum / num_samples,
        "exact_match": exact_sum / num_samples,
    }


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    encoder: FormulaDenseNetEncoder,
    decoder: FormulaTransformerDecoder,
    device: torch.device,
) -> None:
    """Load encoder/decoder weights from a training checkpoint."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])


def run_evaluation(
    *,
    checkpoint_path: Union[str, Path],
    test_loader: DataLoader,
    vocab: Vocab,
    encoder,
    decoder,
    hidden_dim: int = 512,
    max_gen_len: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Build models, load ``best.pt``, and evaluate on ``test_loader``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    load_checkpoint(checkpoint_path, encoder, decoder, device)

    metrics = evaluate_test_set(
        encoder=encoder,
        decoder=decoder,
        test_loader=test_loader,
        decode_fn=make_decode_fn(vocab),
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        device=device,
        max_gen_len=max_gen_len,
    )

    print(f"Test CER: {metrics['cer']:.4f}")
    print(f"Test BLEU: {metrics['bleu']:.4f}")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.4f}")
    return metrics


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Autoregressive test evaluation")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="IM2LATEX dataset root directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to best.pt checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    vocab = build_vocab_from_train(root)

    test_dataset = LatexDataset(
        root,
        split="test",
        vocab=vocab,
        image_postprocess=gray_normalize_neg1_pos1(),
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        max_gen_len=args.max_gen_len,
    )


if __name__ == "__main__":
    main()
