"""
Neural network building blocks for Image-to-LaTeX formula recognition.

- ``FormulaDenseNetEncoder``: maps grayscale images to a memory sequence.
- ``FormulaTransformerDecoder``: autoregressively decodes LaTeX tokens from memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights, densenet121


class LearnedPositionalEncoding2D(nn.Module):
    """
  Learnable 2D positional bias added to convolutional feature maps.

  A separate embedding is learned for each spatial location, preserving
  the row/column structure of the formula before flattening to a sequence.
  """

    def __init__(self, num_channels: int, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        # Shape: [1, C, H, W] — broadcast over batch in forward.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_channels, height, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps, shape [B, C, H, W].

        Returns:
            x with positional bias added, shape [B, C, H, W].
        """
        _, _, h, w = x.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"Positional encoding expects spatial size ({self.height}, {self.width}), "
                f"got ({h}, {w})."
            )
        return x + self.pos_embed


class FormulaDenseNetEncoder(nn.Module):
    """
    DenseNet-121 encoder for grayscale mathematical formula images.

    Pipeline (tensor shapes use default input 128×512):
        1. Backbone features     : [B, 1, 128, 512]  →  [B, 1024, H', W']
        2. Bottleneck projection : [B, 1024, H', W'] →  [B, hidden_dim, H', W']
        3. 2D positional encoding (residual)
        4. Sequence flatten      : [B, hidden_dim, H', W'] →  [B, H'·W', hidden_dim]

    The global average pooling and classifier head of DenseNet-121 are not used;
    only convolutional feature maps are returned for the decoder.
    """

    DENSENET_OUT_CHANNELS = 1024

    def __init__(
        self,
        hidden_dim: int = 512,
        pretrained: bool = False,
        input_height: int = 128,
        input_width: int = 512,
    ) -> None:
        """
        Args:
            hidden_dim: Output channel / embedding size (decoder d_model).
            pretrained: If True, load ImageNet weights into the backbone
                (first conv is re-initialized for 1-channel input).
            input_height: Expected image height for positional-encoding sizing.
            input_width: Expected image width for positional-encoding sizing.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_height = input_height
        self.input_width = input_width

        # ------------------------------------------------------------------
        # Step 1: DenseNet-121 feature extractor (no pool / classifier)
        # ------------------------------------------------------------------
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features

        # Replace RGB stem with a single-channel stem for grayscale input.
        self._adapt_first_convolution_to_grayscale()

        # ------------------------------------------------------------------
        # Step 2: 1×1 conv bottleneck — 1024 → hidden_dim
        # ------------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                self.DENSENET_OUT_CHANNELS,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Step 3: Learned 2D positional encoding (size from a dry-run forward)
        # ------------------------------------------------------------------
        feat_h, feat_w = self._infer_feature_map_size(input_height, input_width)
        self.pos_encoding = LearnedPositionalEncoding2D(
            num_channels=hidden_dim,
            height=feat_h,
            width=feat_w,
        )
        self._feat_height = feat_h
        self._feat_width = feat_w

    def _adapt_first_convolution_to_grayscale(self) -> None:
        """Replace features.conv0 (3→64) with a 1-channel equivalent."""
        old_conv: nn.Conv2d = self.features.conv0  # type: ignore[assignment]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        self.features.conv0 = new_conv

    @torch.no_grad()
    def _infer_feature_map_size(self, height: int, width: int) -> tuple[int, int]:
        """Run a dummy tensor through the backbone to get (H', W')."""
        dummy = torch.zeros(1, 1, height, width)
        features = self._extract_backbone_features(dummy)
        return int(features.shape[2]), int(features.shape[3])

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        DenseNet-121 convolutional trunk only (no GAP, no FC).

        Args:
            x: [B, 1, H, W]

        Returns:
            Feature maps [B, 1024, H', W'].
        """
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of formula images into a sequence for a Transformer decoder.

        Args:
            x: Grayscale images, shape [B, 1, 128, 512] (H, W configurable at init).

        Returns:
            Memory sequence, shape [B, Seq_Len, hidden_dim] where
            Seq_Len = H' × W' (spatial dims flattened in row-major order).
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(
                f"Expected input shape [B, 1, H, W], got {tuple(x.shape)}."
            )

        batch_size = x.size(0)

        # --- Backbone ---
        # [B, 1, H, W] → [B, 1024, H', W']
        features = self._extract_backbone_features(x)

        # --- Bottleneck ---
        # [B, 1024, H', W'] → [B, hidden_dim, H', W']
        projected = self.bottleneck(features)

        # --- 2D positional encoding (preserves spatial layout before flatten) ---
        # [B, hidden_dim, H', W'] → [B, hidden_dim, H', W']
        encoded = self.pos_encoding(projected)

        # --- Sequence conversion: flatten H' and W' into Seq_Len ---
        # [B, hidden_dim, H', W'] → [B, hidden_dim, H'*W'] → [B, Seq_Len, hidden_dim]
        seq_len = encoded.size(2) * encoded.size(3)
        sequence = encoded.flatten(2).transpose(1, 2).contiguous()
        # sequence shape: [B, Seq_Len, hidden_dim]

        assert sequence.shape == (batch_size, seq_len, self.hidden_dim), (
            f"Unexpected output shape {tuple(sequence.shape)}; "
            f"expected ({batch_size}, {seq_len}, {self.hidden_dim})."
        )

        return sequence

    @property
    def sequence_length(self) -> int:
        """Number of tokens produced for the configured input resolution."""
        return self._feat_height * self._feat_width

    @property
    def feature_map_size(self) -> tuple[int, int]:
        """Spatial size (H', W') of backbone output before flattening."""
        return self._feat_height, self._feat_width


class LearnedPositionalEncoding1D(nn.Module):
    """
    Learnable 1D positional bias for decoder token sequences.

    Each position in the target sequence receives a dedicated embedding so the
    decoder can distinguish token order during self-attention.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        # Shape: [1, max_len, d_model] — broadcast over batch in forward.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings, shape [B, T, d_model].

        Returns:
            Embeddings with positional bias, shape [B, T, d_model].
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}."
            )
        return x + self.pos_embed[:, :seq_len, :]


class FormulaTransformerDecoder(nn.Module):
    """
    Transformer decoder for autoregressive LaTeX token prediction.

    Pipeline (tensor shapes):
        1. Token embedding     : [B, T]              →  [B, T, hidden_dim]
        2. 1D positional enc.  : [B, T, hidden_dim]  →  [B, T, hidden_dim]
        3. Self-attention      : causal mask on tgt;  [B, T, hidden_dim]
        4. Cross-attention     : attends to memory   [B, S, hidden_dim]
        5. Output projection   : [B, T, hidden_dim]  →  [B, T, vocab_size]

    ``memory`` is the encoder output (S = encoder Seq_Len, e.g. 64 for 128×512 images).
    ``tgt`` is the shifted-right target token IDs (teacher forcing during training).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 512,
        padding_idx: int = 0,
    ) -> None:
        """
        Args:
            vocab_size: Size of the LaTeX token vocabulary (output logits).
            hidden_dim: Model dimension (d_model); must match ``FormulaDenseNetEncoder``.
            nhead: Number of attention heads.
            num_layers: Number of stacked ``TransformerDecoderLayer`` blocks.
            dim_feedforward: FFN inner dimension.
            dropout: Dropout probability inside the decoder stack.
            max_tgt_len: Maximum target sequence length for 1D positional encoding.
            padding_idx: Embedding index reserved for padding tokens.
        """
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})."
            )

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_tgt_len = max_tgt_len
        self.padding_idx = padding_idx

        # ------------------------------------------------------------------
        # Step 1: Token embedding (vocab → d_model)
        # ------------------------------------------------------------------
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )

        # ------------------------------------------------------------------
        # Step 2: 1D learned positional encoding (word order)
        # ------------------------------------------------------------------
        self.pos_encoding = LearnedPositionalEncoding1D(
            d_model=hidden_dim,
            max_len=max_tgt_len,
        )

        # ------------------------------------------------------------------
        # Step 3–4: Transformer decoder (self-attn + cross-attn to memory)
        # ------------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # ------------------------------------------------------------------
        # Step 5: Linear projection to vocabulary logits
        # ------------------------------------------------------------------
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for projection; standard for embedding is default."""
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def generate_causal_mask(
        tgt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a look-ahead (causal) mask for autoregressive decoding.

        Positions may attend only to themselves and earlier tokens.
        PyTorch expects a bool mask of shape [T, T] where True means *masked*
        (cannot attend).

        Args:
            tgt_len: Target sequence length T.
            device: Device for the mask tensor.

        Returns:
            Causal mask, shape [T, T].
        """
        return torch.triu(
            torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode LaTeX tokens conditioned on encoder memory.

        Args:
            tgt: Target token IDs, shape [B, Tgt_Seq_Len] (e.g. BOS + tokens).
            memory: Encoder output, shape [B, Seq_Len, hidden_dim].
            tgt_key_padding_mask: Optional padding mask for tgt, shape [B, T]
                with True at padded positions.
            memory_key_padding_mask: Optional padding mask for memory, shape [B, S].

        Returns:
            Logits over the vocabulary, shape [B, Tgt_Seq_Len, vocab_size].
        """
        if tgt.dim() != 2:
            raise ValueError(f"Expected tgt shape [B, T], got {tuple(tgt.shape)}.")
        if memory.dim() != 3:
            raise ValueError(
                f"Expected memory shape [B, S, hidden_dim], got {tuple(memory.shape)}."
            )
        if memory.size(-1) != self.hidden_dim:
            raise ValueError(
                f"memory hidden dim {memory.size(-1)} != decoder hidden_dim "
                f"{self.hidden_dim}."
            )

        batch_size, tgt_len = tgt.shape
        device = tgt.device

        # --- Embedding: [B, T] → [B, T, hidden_dim] ---
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.hidden_dim)

        # --- 1D positional encoding: [B, T, hidden_dim] → [B, T, hidden_dim] ---
        tgt_emb = self.pos_encoding(tgt_emb)

        # --- Causal mask for self-attention: [T, T] ---
        tgt_mask = self.generate_causal_mask(tgt_len, device)

        # --- TransformerDecoder ---
        # Self-attention (masked) + cross-attention to memory:
        #   tgt:    [B, T, hidden_dim]
        #   memory: [B, Seq_Len, hidden_dim]
        #   out:    [B, T, hidden_dim]
        decoder_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # --- Projection to vocabulary: [B, T, hidden_dim] → [B, T, vocab_size] ---
        logits = self.output_projection(decoder_out)

        assert logits.shape == (batch_size, tgt_len, self.vocab_size), (
            f"Unexpected logits shape {tuple(logits.shape)}; "
            f"expected ({batch_size}, {tgt_len}, {self.vocab_size})."
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding conditioned on encoder memory.

        Args:
            memory: Encoder output, shape [B, Seq_Len, hidden_dim].
            sos_id: Start-of-sequence token id.
            eos_id: End-of-sequence token id.
            max_len: Maximum generated sequence length (including SOS).

        Returns:
            Generated token IDs, shape [B, generated_length].
        """
        if memory.dim() != 3:
            raise ValueError(
                f"Expected memory shape [B, S, hidden_dim], got {tuple(memory.shape)}."
            )

        batch_size = memory.size(0)
        device = memory.device

        generated = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        steps = min(max_len, self.max_tgt_len) - 1
        for _ in range(steps):
            logits = self.forward(generated, memory)
            next_ids = logits[:, -1, :].argmax(dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )
            finished = finished | (next_ids == eos_id)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return generated



# ============================================================
# ResNet34 Encoder
# ============================================================

class FormulaResNetEncoder(nn.Module):
    """
    CNN encoder using ResNet-34 backbone.

    Input:
        [B, 1, 128, 512]

    Output:
        [B, Seq_Len, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        backbone = models.resnet34(pretrained=pretrained)

        # Convert RGB stem -> grayscale stem
        backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Remove avgpool + classifier
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # [B, 1, 128, 512]
        features = self.cnn(x)

        # [B, 512, H', W']
        batch_size, channels, height, width = features.size()

        # [B, 512, H'*W']
        features = features.view(
            batch_size,
            channels,
            height * width,
        )

        # [B, H'*W', 512]
        features = features.permute(0, 2, 1).contiguous()

        return features


# ============================================================
# Attention Module
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        enc_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 256,
    ) -> None:
        super().__init__()

        self.enc_attn = nn.Linear(enc_dim, attn_dim)
        self.dec_attn = nn.Linear(dec_dim, attn_dim)

        self.full_attn = nn.Linear(attn_dim, 1)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # encoder_out:
        # [B, Seq_Len, enc_dim]

        # decoder_hidden:
        # [B, dec_dim]

        att1 = self.enc_attn(encoder_out)

        att2 = self.dec_attn(decoder_hidden).unsqueeze(1)

        att = self.full_attn(
            torch.tanh(att1 + att2)
        ).squeeze(2)

        alpha = F.softmax(att, dim=1)

        context = (
            encoder_out * alpha.unsqueeze(2)
        ).sum(dim=1)

        return context, alpha


# ============================================================
# LSTM Attention Decoder
# ============================================================

class FormulaLSTMAttentionDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        enc_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 256,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dec_dim = dec_dim

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
        )

        self.attention = Attention(
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
        )

        self.rnn = nn.LSTMCell(
            emb_dim + enc_dim,
            dec_dim,
        )

        self.fc = nn.Linear(
            dec_dim,
            vocab_size,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask=None,
    ) -> torch.Tensor:
        """
        Teacher-forced decoding.

        Args:
            tgt:
                [B, T]

            memory:
                [B, Seq_Len, enc_dim]

        Returns:
            logits:
                [B, T, vocab_size]
        """

        batch_size = memory.size(0)
        seq_len = tgt.size(1)

        device = memory.device

        h = torch.zeros(
            batch_size,
            self.dec_dim,
            device=device,
        )

        c = torch.zeros(
            batch_size,
            self.dec_dim,
            device=device,
        )

        predictions = torch.zeros(
            batch_size,
            seq_len,
            self.vocab_size,
            device=device,
        )

        for t in range(seq_len):

            embeddings = self.embedding(
                tgt[:, t]
            )

            context, _ = self.attention(
                memory,
                h,
            )

            h, c = self.rnn(
                torch.cat(
                    [embeddings, context],
                    dim=1,
                ),
                (h, c),
            )

            preds = self.fc(h)

            predictions[:, t, :] = preds

        return predictions

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.
        """

        batch_size = memory.size(0)

        device = memory.device

        h = torch.zeros(
            batch_size,
            self.dec_dim,
            device=device,
        )

        c = torch.zeros(
            batch_size,
            self.dec_dim,
            device=device,
        )

        generated = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_len - 1):

            current_token = generated[:, -1]

            embeddings = self.embedding(
                current_token
            )

            context, _ = self.attention(
                memory,
                h,
            )

            h, c = self.rnn(
                torch.cat(
                    [embeddings, context],
                    dim=1,
                ),
                (h, c),
            )

            logits = self.fc(h)

            next_ids = logits.argmax(dim=-1)

            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )

            finished = finished | (
                next_ids == eos_id
            )

            generated = torch.cat(
                [
                    generated,
                    next_ids.unsqueeze(1),
                ],
                dim=1,
            )

            if finished.all():
                break

        return generated



# ============================================================
# Lightweight CNN Encoder
# ============================================================

class ConvBNAct(nn.Module):

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                cin,
                cout,
                kernel_size=k,
                stride=s,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBNAct(
            channels,
            channels,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.act(
            self.conv2(
                self.conv1(x)
            ) + x
        )


class FormulaCNNEncoder(nn.Module):
    """
    Lightweight CNN encoder with learned 2D positional embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.net = nn.Sequential(

            # 128x512 -> 64x256
            ConvBNAct(1, 64),
            ConvBNAct(64, 64),
            nn.MaxPool2d(2, 2),

            # -> 32x128
            ConvBNAct(64, 128),
            ResBlock(128),
            nn.MaxPool2d(2, 2),

            # -> 16x64
            ConvBNAct(128, 256),
            ResBlock(256),
            nn.MaxPool2d(2, 2),

            # -> 8x64
            ConvBNAct(256, 256),
            ResBlock(256),
            nn.MaxPool2d((2, 1), (2, 1)),

            ConvBNAct(256, d_model),
            ResBlock(d_model),
        )

        self.row_embed = nn.Parameter(
            torch.randn(32, d_model) * 0.02
        )

        self.col_embed = nn.Parameter(
            torch.randn(128, d_model) * 0.02
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, D, H, W]
        x = self.net(x)

        batch_size, d_model, height, width = x.shape

        # [B, H, W, D]
        x = x.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        pos = (
            self.row_embed[:height].unsqueeze(1)
            +
            self.col_embed[:width].unsqueeze(0)
        )

        x = x + pos.unsqueeze(0)

        # [B, H*W, D]
        x = x.view(
            batch_size,
            height * width,
            d_model,
        )

        return self.norm(x)


# ============================================================
# Sinusoidal Positional Encoding
# ============================================================

class TokenPositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
    ) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(max_len).float().unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return x + self.pe[:, :x.size(1)]


# ============================================================
# Causal Mask Utility
# ============================================================

def causal_mask(
    size: int,
    device: torch.device,
) -> torch.Tensor:

    return torch.triu(
        torch.ones(
            size,
            size,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


# ============================================================
# Transformer Decoder V2
# ============================================================

class FormulaTransformerDecoderV2(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff: int = 1024,
        max_len: int = 256,
    ) -> None:
        super().__init__()

        self.pad_id = pad_id
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.embed = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.pos = TokenPositionalEncoding(
            d_model,
            max_len=max_len,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.out = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt:
                [B, T]

            memory:
                [B, Seq_Len, d_model]
        """

        tgt_pad_mask = (
            tgt == self.pad_id
        )

        tgt_mask = causal_mask(
            tgt.size(1),
            tgt.device,
        )

        tgt_emb = (
            self.embed(tgt)
            * math.sqrt(self.d_model)
        )

        tgt_emb = self.pos(tgt_emb)

        x = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        x = self.norm(x)

        logits = self.out(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:

        batch_size = memory.size(0)

        device = memory.device

        generated = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        steps = min(
            max_len,
            self.max_len,
        ) - 1

        for _ in range(steps):

            logits = self.forward(
                generated,
                memory,
            )

            next_ids = logits[:, -1].argmax(dim=-1)

            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )

            finished = finished | (
                next_ids == eos_id
            )

            generated = torch.cat(
                [
                    generated,
                    next_ids.unsqueeze(1),
                ],
                dim=1,
            )

            if finished.all():
                break

        return generated



# ============================================================
# Basic CNN Encoder
# ============================================================

class FormulaBasicCNNEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(

            # [B, 1, 128, 512]
            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # -> [B, 64, 64, 256]

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # -> [B, 128, 32, 128]

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),

            # -> [B, 256, 32, 128]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            memory:
                [B, Seq_Len, 256]
        """

        x = self.conv(x)

        batch_size, channels, height, width = x.size()

        # [B, 256, H*W]
        x = x.view(
            batch_size,
            channels,
            height * width,
        )

        # [B, H*W, 256]
        x = x.permute(
            0,
            2,
            1,
        ).contiguous()

        return x


# ============================================================
# GRU Decoder 
# ============================================================

class FormulaGRUDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.rnn = nn.GRU(
            embed_dim + hidden_dim,
            hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

        # encoder feature projection
        self.encoder_feature_to_hidden = nn.Linear(
            encoder_dim,
            hidden_dim,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced decoding.

        Args:
            tgt:
                [B, T]

            memory:
                [B, Seq_Len, encoder_dim]

        Returns:
            logits:
                [B, T, vocab_size]
        """

        batch_size = tgt.size(0)

        target_seq_len = tgt.size(1)

        # global average context
        mean_memory = torch.mean(
            memory,
            dim=1,
        )

        context = self.encoder_feature_to_hidden(
            mean_memory
        )

        embedded_targets = self.embedding(
            tgt
        )

        repeated_context = context.unsqueeze(1).repeat(
            1,
            target_seq_len,
            1,
        )

        rnn_input = torch.cat(
            [
                embedded_targets,
                repeated_context,
            ],
            dim=2,
        )

        initial_hidden = torch.zeros(
            1,
            batch_size,
            self.hidden_dim,
            device=tgt.device,
        )

        rnn_output, _ = self.rnn(
            rnn_input,
            initial_hidden,
        )

        predictions = self.fc(
            rnn_output
        )

        return predictions

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.
        """

        batch_size = memory.size(0)

        device = memory.device

        mean_memory = torch.mean(
            memory,
            dim=1,
        )

        context = self.encoder_feature_to_hidden(
            mean_memory
        )

        hidden = torch.zeros(
            1,
            batch_size,
            self.hidden_dim,
            device=device,
        )

        generated = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_len - 1):

            current_token = generated[:, -1]

            embedded = self.embedding(
                current_token
            ).unsqueeze(1)

            repeated_context = context.unsqueeze(1)

            rnn_input = torch.cat(
                [
                    embedded,
                    repeated_context,
                ],
                dim=2,
            )

            rnn_output, hidden = self.rnn(
                rnn_input,
                hidden,
            )

            logits = self.fc(
                rnn_output.squeeze(1)
            )

            next_ids = logits.argmax(dim=-1)

            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )

            finished = finished | (
                next_ids == eos_id
            )

            generated = torch.cat(
                [
                    generated,
                    next_ids.unsqueeze(1),
                ],
                dim=1,
            )

            if finished.all():
                break

        return generated



# ============================================================
# Basic CNN Encoder V2
# ============================================================

class FormulaBasicCNNEncoderV2(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # 128x512 -> 64x256

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # 64x256 -> 32x128

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            memory:
                [B, Seq_Len, 256]
        """

        x = self.conv(x)

        batch_size, channels, height, width = x.size()

        # [B, 256, H*W]
        x = x.view(
            batch_size,
            channels,
            height * width,
        )

        # [B, H*W, 256]
        x = x.permute(
            0,
            2,
            1,
        ).contiguous()

        return x


# ============================================================
# GRU Decoder V2
# ============================================================

class FormulaGRUDecoderV2(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.rnn = nn.GRU(
            embed_dim + encoder_dim,
            hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced decoding.

        Args:
            tgt:
                [B, T]

            memory:
                [B, Seq_Len, encoder_dim]
        """

        embedded = self.embedding(
            tgt
        )

        # global average context
        context = torch.mean(
            memory,
            dim=1,
            keepdim=True,
        )

        context = context.repeat(
            1,
            tgt.size(1),
            1,
        )

        rnn_input = torch.cat(
            [
                embedded,
                context,
            ],
            dim=-1,
        )

        rnn_output, _ = self.rnn(
            rnn_input
        )

        logits = self.fc(
            rnn_output
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:

        batch_size = memory.size(0)

        device = memory.device

        context = torch.mean(
            memory,
            dim=1,
            keepdim=True,
        )

        hidden = None

        generated = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_len - 1):

            current_token = generated[:, -1]

            embedded = self.embedding(
                current_token
            ).unsqueeze(1)

            rnn_input = torch.cat(
                [
                    embedded,
                    context,
                ],
                dim=-1,
            )

            rnn_output, hidden = self.rnn(
                rnn_input,
                hidden,
            )

            logits = self.fc(
                rnn_output.squeeze(1)
            )

            next_ids = logits.argmax(dim=-1)

            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )

            finished = finished | (
                next_ids == eos_id
            )

            generated = torch.cat(
                [
                    generated,
                    next_ids.unsqueeze(1),
                ],
                dim=1,
            )

            if finished.all():
                break

        return generated