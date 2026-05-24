"""
IM2LATEX-100k integrated data pipeline: parsing, vocab, image preprocessing,
visual comparison, PyTorch Dataset/DataLoader.

Expected dataset layout (CSV-based):

    <root>/
      data/
        im2latex_train.csv           # header includes image path + LaTeX string
        im2latex_validate.csv
        im2latex_test.csv
        images/                      # rendered formula images
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Iterable, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

# -----------------------------------------------------------------------------
# 1. Data parsing
# -----------------------------------------------------------------------------


def parse_split_file(split_path: str | Path) -> list[tuple[str, str]]:
    """
    Parse split CSV into (image_relative_path, latex_formula_string).
    Header row is required and skipped by DictReader.
    Supports swapped column order by resolving columns by header names.
    """
    path = Path(split_path)
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        if not reader.fieldnames:
            return pairs

        fields = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
        image_col = (
            fields.get("image")
            or fields.get("image_path")
            or fields.get("path")
            or fields.get("filename")
            or fields.get("file")
        )
        formula_col = fields.get("formula") or fields.get("latex")

        if image_col is None or formula_col is None:
            raise ValueError(
                f"Could not infer image/formula columns in split CSV: {path}. "
                f"Found headers: {reader.fieldnames}"
            )

        for row in reader:
            img_rel = (row.get(image_col) or "").strip()
            formula_text = (row.get(formula_col) or "").strip()
            if not img_rel or not formula_text:
                continue
            pairs.append((img_rel, formula_text))
    return pairs


# -----------------------------------------------------------------------------
# Vocab: regex LaTeX tokenizer + specials
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = ("<pad>", "<sos>", "<eos>", "<unk>")

# LaTeX commands as single tokens; braces, numbers, and non-whitespace chunks.
LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+\*?"  # \frac, \sqrt*, etc.
    r"|\\[^a-zA-Z\s]"  # \, \; \(single-char control seq after \)
    r"|\{|\}"
    r"|[0-9]+"
    r"|[a-zA-Z]+"
    r"|[^\s\\\{\}a-zA-Z0-9]"
)


def tokenize_latex(formula: str) -> list[str]:
    """Tokenize a LaTeX string; commands like \\frac are single tokens."""
    s = formula.strip()
    if not s:
        return []
    tokens: list[str] = []
    for m in LATEX_TOKEN_RE.finditer(s):
        tok = m.group(0)
        if tok.strip() == "":
            continue
        tokens.append(tok)
    return tokens


class Vocab:
    """
    Vocabulary over LaTeX token strings with fixed special token ids at the start.
    """

    def __init__(
        self,
        token_to_idx: dict[str, int],
        idx_to_token: dict[int, str] | None = None,
    ) -> None:
        self.stoi = dict(token_to_idx)
        if idx_to_token is None:
            self.itos = {i: t for t, i in self.stoi.items()}
        else:
            self.itos = dict(idx_to_token)

    @classmethod
    def from_corpus(
        cls,
        formulas: Iterable[str],
        min_freq: int = 1,
    ) -> "Vocab":
        """Build vocab from raw LaTeX strings; specials first, then by frequency."""
        from collections import Counter

        counts: Counter[str] = Counter()
        for f in formulas:
            for tok in tokenize_latex(f):
                counts[tok] += 1

        token_to_idx: dict[str, int] = {}
        for i, sp in enumerate(SPECIAL_TOKENS):
            token_to_idx[sp] = i

        next_id = len(SPECIAL_TOKENS)
        for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if c < min_freq or tok in token_to_idx:
                continue
            token_to_idx[tok] = next_id
            next_id += 1

        idx_to_token = {v: k for k, v in token_to_idx.items()}
        return cls(token_to_idx, idx_to_token)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def sos_id(self) -> int:
        return self.stoi["<sos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def encode(
        self,
        formula: str,
        *,
        add_sos_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize_latex(formula)]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            ids = ids[: max_len]
            ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        parts: list[str] = []
        skip_ids = {self.pad_id, self.sos_id, self.eos_id}
        for i in ids:
            if skip_special and i in skip_ids:
                continue
            parts.append(self.itos.get(i, "<unk>"))
        return " ".join(parts)

    def __len__(self) -> int:
        return len(self.stoi)


# -----------------------------------------------------------------------------
# 2. Image preprocessing (grayscale vs Otsu binarize; padding-first resize)
# -----------------------------------------------------------------------------

ImageMode = Literal["original", "binarized_otsu", "binarized_adaptive"]


def preprocess_formula_image(
    image_bgr_or_gray: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    mode: ImageMode = "original",
    pad_value: float = 1.0,
) -> np.ndarray:
    """
    Preprocess a single formula image.

    - Converts to grayscale if input is BGR.
    - Optional Otsu binarization (255 foreground / 0 background style -> [0,1]).
    - Padding-first resize: scale to fit inside (target_h, target_w) preserving
      aspect ratio, then letterbox-pad to exact size (no stretching).
    - Output float32 tensor shaped (1, target_h, target_w) with values in [0, 1].

    Parameters
    ----------
    pad_value :
        Value for padded regions in [0, 1] space (1.0 = white background typical for rendered formulas).
    """
    if image_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray.astype(np.uint8)

    h, w = gray.shape[:2]

    if mode == "binarized_otsu":
        # Otsu on 8-bit grayscale
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Normalize to [0, 1]; ink as high or low depends on inversion — keep as uint8-like then float
        work = (bw.astype(np.float32) / 255.0).astype(np.float32)
    elif mode == "binarized_adaptive":
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        bw = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 20
        )
        work = (bw.astype(np.float32) / 255.0) 
    elif mode == "original":
        work = (gray.astype(np.float32) / 255.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Padding-first: resize to fit inside box, then pad
    scale = min(target_h / float(h), target_w / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(work, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w), pad_value, dtype=np.float32)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    y1 = y0 + new_h
    x1 = x0 + new_w
    canvas[y0:y1, x0:x1] = np.clip(resized, 0.0, 1.0)

    # Channel dimension for ConvNets
    return np.expand_dims(canvas, axis=0)


def load_image_path(path: str | Path) -> np.ndarray:
    """Read image with OpenCV (BGR or gray); raises if unreadable."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return arr


# -----------------------------------------------------------------------------
# 3. Visual comparison tool
# -----------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    root: Path
    train_split: Path
    val_split: Path
    test_split: Path
    image_root: Path
    target_h: int = 128
    target_w: int = 512
    pad_value: float = 1.0


def gray_normalize_neg1_pos1() -> Normalize:
    """torchvision: map [0, 1] single-channel inputs to roughly [-1, 1]."""
    return Normalize(mean=(0.5,), std=(0.5,))


def is_running_on_colab() -> bool:
    """
    Detect Google Colab runtime using the requested notebook-style check:
    'google.colab' in str(get_ipython()).
    """
    try:
        ipy = get_ipython()  # type: ignore[name-defined]
    except Exception:
        return False
    if ipy is None:
        return False
    return "google.colab" in str(ipy)


def default_num_workers(num_workers: int | None = None) -> int:
    """
    Environment-aware worker count.
    - Colab: use 4 workers (good default for /content local disk)
    - Windows local: default 0 for safety
    - Other local: up to 4 workers
    """
    if num_workers is not None:
        return num_workers
    if is_running_on_colab():
        return 4
    if os.name == "nt":
        return 0
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu))


def default_config(root: str | Path) -> PipelineConfig:
    r = Path(root)
    return PipelineConfig(
        root=r,
        train_split=r / "data" / "im2latex_train.csv",
        val_split=r / "data" / "im2latex_validate.csv",
        test_split=r / "data" / "im2latex_test.csv",
        image_root=r / "data" / "images",
    )


def compare_preprocessing(
    root: str | Path,
    n_samples: int = 5,
    *,
    split: Literal["train", "val", "test"] = "train",
    seed: int | None = 42,
    target_h: int = 128,
    target_w: int = 512,
    pad_value: float = 1.0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Side-by-side matplotlib figure: Original (grayscale path) vs Binarized (Otsu)
    for randomly sampled images from a split list.
    """
    cfg = default_config(root)
    split_paths = {
        "train": cfg.train_split,
        "val": cfg.val_split,
        "test": cfg.test_split,
    }
    sp_path = split_paths[split]
    if not sp_path.is_file():
        raise FileNotFoundError(f"Split file not found: {sp_path}")

    pairs = parse_split_file(sp_path)
    if not pairs:
        raise ValueError(f"No entries in {sp_path}")

    rng = random.Random(seed)
    picks = rng.sample(pairs, k=min(n_samples, len(pairs)))
    n_rows = len(picks)

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 2.5 * max(n_rows, 1)))
    if n_rows == 1:
        axes = np.array([axes])

    last_dir: Path | None = None
    for i, (rel_img, _) in enumerate(picks):
        normalized_rel = rel_img.replace("/", os.sep)
        img_path = cfg.image_root / normalized_rel
        if not img_path.is_file():
            alt = cfg.root / normalized_rel
            if alt.is_file():
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path} (or {alt})")

        last_dir = img_path.parent
        raw = load_image_path(img_path)
        orig = preprocess_formula_image(
            raw,
            target_h=target_h,
            target_w=target_w,
            mode="original",
            pad_value=pad_value,
        )
        binary_otsu = preprocess_formula_image(
            raw,
            target_h=target_h,
            target_w=target_w,
            mode="binarized_otsu",
            pad_value=pad_value,
        )
        binary_adaptive = preprocess_formula_image(
            raw,
            target_h=target_h,
            target_w=target_w,
            mode="binarized_adaptive",
            pad_value=pad_value,
        )

        for ax, tensor, title in zip(
            axes[i],
            (orig, binary_otsu, binary_adaptive),
            ("Original (grayscale)", "Binarized (Otsu)", "Binarized (Adaptive)"),
        ):
            ax.imshow(tensor[0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle(
        f"{split}: preprocessing comparison"
        + (f" (sample dir: {last_dir})" if last_dir is not None else "")
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# 4. PyTorch Dataset & DataLoader
# -----------------------------------------------------------------------------


class LatexDataset(Dataset):
    """
    IM2LATEX-style dataset: split CSV + images on disk.

    `image_mode`: 'original' | 'binarized' — use the same preprocessing for all items.

    Optional `image_postprocess` (e.g. ``torchvision.transforms.Normalize``) runs
    after OpenCV preprocessing for seamless integration with torchvision models.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"],
        vocab: Vocab,
        *,
        image_mode: ImageMode = "original",
        target_h: int = 128,
        target_w: int = 512,
        pad_value: float = 1.0,
        max_formula_len: int | None = None,
        image_postprocess: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        cfg = default_config(root)
        split_paths = {
            "train": cfg.train_split,
            "val": cfg.val_split,
            "test": cfg.test_split,
        }
        self.pairs = parse_split_file(split_paths[split])
        self.vocab = vocab
        self.image_mode = image_mode
        self.target_h = target_h
        self.target_w = target_w
        self.pad_value = pad_value
        self.max_formula_len = max_formula_len
        self.image_postprocess = image_postprocess

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve_image(self, rel_img: str) -> Path:
        cfg = default_config(self.root)
        normalized_rel = rel_img.replace("/", os.sep)
        p = cfg.image_root / normalized_rel
        if p.is_file():
            return p
        alt = self.root / normalized_rel
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"Image not found: {p} (tried {alt})")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[int]]:
        rel_img, text = self.pairs[idx]
        path = self._resolve_image(rel_img)
        raw = load_image_path(path)
        img = preprocess_formula_image(
            raw,
            target_h=self.target_h,
            target_w=self.target_w,
            mode=self.image_mode,
            pad_value=self.pad_value,
        )
        img_t = torch.from_numpy(img).float()
        if self.image_postprocess is not None:
            img_t = self.image_postprocess(img_t)

        token_ids = self.vocab.encode(text, add_sos_eos=True, max_len=self.max_formula_len)

        return {
            "image": img_t,
            "formula_ids": token_ids,
            "formula_str": text,
        }


def collate_formulas(
    batch: list[dict],
    *,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length token-id sequences; stack images [B, 1, H, W]."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    seqs = [torch.tensor(b["formula_ids"], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    return {
        "images": images,
        "formula_ids": padded,
        "lengths": lengths,
    }


def make_collate_fn(pad_id: int):
    def _fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_formulas(batch, pad_id=pad_id)

    return _fn


def build_vocab_from_train(
    root: str | Path,
    train_split: str | Path | None = None,
) -> Vocab:
    """Collect all formulas from train split and build Vocab."""
    r = Path(root)
    if train_split is None:
        train_split = r / "data" / "im2latex_train.csv"

    pairs = parse_split_file(train_split)
    seen = [formula for _, formula in pairs]
    return Vocab.from_corpus(seen)


def create_dataloader(
    dataset: LatexDataset,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Training-ready DataLoader. On Windows, num_workers=0 is reliable unless you
    guard with ``if __name__ == '__main__':``. Increase num_workers on Linux for throughput.
    """
    pad_id = dataset.vocab.pad_id
    workers = default_num_workers(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=make_collate_fn(pad_id),
    )


# -----------------------------------------------------------------------------
# Main: demo comparison + vocab size
# -----------------------------------------------------------------------------


def _main() -> None:
    import argparse

    on_colab = is_running_on_colab()
    default_root = "/content/dataset" if on_colab else os.environ.get("IM2LATEX_ROOT", ".")

    parser = argparse.ArgumentParser(description="IM2LATEX data pipeline demo")
    parser.add_argument(
        "--root",
        type=str,
        default=default_root,
        help="Dataset root (Colab default: /content/dataset; Local default: IM2LATEX_ROOT or .)",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() in compare (save only if --save)")
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save comparison figure PNG",
    )
    args = parser.parse_args()
    root = Path(args.root)

    vocab = build_vocab_from_train(root)
    print(f"Vocabulary size (including special tokens): {len(vocab)}")
    print(f"Environment: {'Colab' if on_colab else 'Local'}")
    print(f"Recommended DataLoader num_workers: {default_num_workers(None)}")

    # Visual verification (thinner strokes): run comparison if split exists
    if not args.no_show or args.save:
        save = args.save or None
        if args.no_show and save is None:
            save = str(root / "preprocess_compare.png")

        compare_preprocessing(
            root,
            n_samples=5,
            split="train",
            save_path=save if save else None,
            show=not args.no_show,
        )


if __name__ == "__main__":
    _main()
