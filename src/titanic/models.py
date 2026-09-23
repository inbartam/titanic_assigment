"""PyTorch model ladder: linear, MLP and attention, on one shared interface.

Three architectures of deliberately increasing capacity, all consuming the
*same* preprocessed arrays and all exposing the same forward signature::

    forward(x_num: FloatTensor[B, n_numeric],
            x_cat: LongTensor[B, n_categorical]) -> FloatTensor[B]   # logits

Returning raw logits rather than probabilities is intentional: it lets the
training loop use :class:`torch.nn.BCEWithLogitsLoss`, which folds the sigmoid
into the loss in a numerically stable way. Callers that want probabilities
apply ``torch.sigmoid`` themselves, exactly once, at inference.

Why a ladder rather than one model: on 712 rows the interesting question is
not which architecture wins but whether they are *distinguishable at all*.
Sharing the preprocessor and the training recipe means any gap between them is
attributable to the architecture and nothing else.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

#: Maximum width of a categorical embedding. With cardinalities of at most 10
#: the rule ``min(8, ceil(card / 2))`` never actually reaches this cap, but it
#: keeps the rule well defined if a higher-cardinality feature is ever added.
MAX_EMBEDDING_DIM = 8


def count_parameters(model: nn.Module) -> int:
    """Count a model's trainable parameters.

    Reported in ``model_config.json`` and shown in the app, because parameter
    count is the honest way to express "how much model is this?" next to a
    validation score from 179 rows.

    Args:
        model: Any torch module.

    Returns:
        Total number of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def embedding_dim(cardinality: int) -> int:
    """Pick an embedding width for a categorical column.

    Uses the common rule of thumb ``min(8, ceil(cardinality / 2))``. With
    cardinalities of 3 to 10 the difference against one-hot encoding is a
    handful of parameters; embeddings are used mainly because they are the
    correct general treatment and they generalise to high-cardinality features.

    Args:
        cardinality: Vocabulary size including the ``<UNK>`` slot.

    Returns:
        Embedding dimension, at least 1.
    """
    return max(1, min(MAX_EMBEDDING_DIM, math.ceil(cardinality / 2)))


class TitanicLinear(nn.Module):
    """Logistic regression, implemented in PyTorch.

    Categorical columns are one-hot encoded and concatenated with the
    standardised numerics, then passed through a single ``Linear(in, 1)``.
    That is logistic regression, but trained with the same loop, loss,
    optimiser, batching and seed as the larger models, so any performance gap
    is attributable to architecture rather than to a different training recipe.
    A scikit-learn ``LogisticRegression`` would not give that guarantee.

    At roughly 25 parameters it is also fully interpretable and trains in
    seconds, which makes it the right default for a dataset of 712 rows.
    """

    def __init__(self, n_numeric: int, cardinalities: list[int]) -> None:
        """Build the linear model.

        Args:
            n_numeric: Number of standardised numeric features.
            cardinalities: Vocabulary size per categorical column.
        """
        super().__init__()
        self.cardinalities = list(cardinalities)

        n_inputs = n_numeric + sum(cardinalities)
        self.linear = nn.Linear(n_inputs, 1)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration sufficient to rebuild this model."""
        return {"type": "fast"}

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Map a batch to survival logits.

        Args:
            x_num: Float tensor ``(B, n_numeric)``.
            x_cat: Long tensor ``(B, n_categorical)`` of vocabulary indices.

        Returns:
            Float tensor ``(B,)`` of logits.
        """
        # One-hot each categorical column separately: the columns have
        # different cardinalities, so they cannot share a single call.
        one_hots = [
            nn.functional.one_hot(x_cat[:, i], num_classes=card).float()
            for i, card in enumerate(self.cardinalities)
        ]
        features = torch.cat([x_num, *one_hots], dim=1)
        # squeeze(-1) turns (B, 1) into (B,) so the loss compares against a
        # (B,) target instead of broadcasting into a (B, B) matrix.
        return self.linear(features).squeeze(-1)


class TitanicMLP(nn.Module):
    """Categorical embeddings feeding a small multi-layer perceptron.

    The required deliverable model. Architecture: one embedding per
    categorical column, concatenated with the numerics, then two hidden layers
    with ReLU and dropout, then a single output unit.

    It is small: roughly 3-4k parameters, about five per
    training row. Anything larger overfits faster without adding capacity that
    712 examples can actually support. Dropout, weight decay and early stopping
    are the three cheap regularisers that matter at this scale.
    """

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        hidden: list[int] | tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        """Build the MLP.

        Args:
            n_numeric: Number of standardised numeric features.
            cardinalities: Vocabulary size per categorical column.
            hidden: Width of each hidden layer.
            dropout: Dropout probability applied after every hidden layer.
        """
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.hidden = list(hidden)
        self.dropout = dropout

        # ModuleList, not a plain list: a plain list would hide the embeddings
        # from .parameters(), so the optimiser would never update them and
        # .to(device) would leave them behind.
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim(card)) for card in self.cardinalities]
        )

        n_inputs = n_numeric + sum(emb.embedding_dim for emb in self.embeddings)
        layers: list[nn.Module] = []
        in_features = n_inputs
        for width in self.hidden:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration sufficient to rebuild this model."""
        return {"type": "deep", "hidden": list(self.hidden), "dropout": self.dropout}

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Map a batch to survival logits.

        Args:
            x_num: Float tensor ``(B, n_numeric)``.
            x_cat: Long tensor ``(B, n_categorical)`` of vocabulary indices.

        Returns:
            Float tensor ``(B,)`` of logits.
        """
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        features = torch.cat([x_num, *embedded], dim=1)
        return self.network(features).squeeze(-1)


class TitanicAttention(nn.Module):
    """A miniature FT-Transformer: self-attention over feature tokens.

    Every feature becomes a token in a shared ``d``-dimensional space
    (categorical columns through an embedding table, numeric columns through a
    learned ``value x vector + bias`` projection). A learned ``[CLS]`` token is
    prepended, and a transformer encoder lets features attend to one another.
    The ``[CLS]`` representation is then read out through a linear head.

    This is the standard modern architecture for tabular deep learning. It is
    included to show it can be implemented correctly and *small* (~7k
    parameters), not because it is expected to win: on 712 rows a more
    expressive model mostly buys higher variance. Reporting that honestly, with
    overlapping confidence intervals, is the point.
    """

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Build the attention model.

        Args:
            n_numeric: Number of standardised numeric features.
            cardinalities: Vocabulary size per categorical column.
            d_model: Token width. Must be divisible by ``n_heads``.
            n_heads: Attention heads per layer.
            n_layers: Number of encoder layers.
            dim_feedforward: Width of each layer's feed-forward block.
            dropout: Dropout inside the encoder layers.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}); "
                "each head takes an equal slice of the token width."
            )

        self.cardinalities = list(cardinalities)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # One token per feature, plus [CLS].
        self.n_tokens = n_numeric + len(self.cardinalities) + 1

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in self.cardinalities]
        )
        # Each numeric feature gets its own Linear(1, d): the token is the
        # scalar value times a learned direction, plus a learned offset. A
        # shared projection would make every numeric feature collinear.
        self.num_projections = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_numeric)])

        # Learned [CLS] token: the slot the head reads from after attention.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Learned per-position embedding. Unlike text, feature order is fixed
        # and meaningful ("column 3 is always Age"), so this is an identity
        # signal rather than a sequence position.
        self.feature_embedding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            # norm_first (pre-LN) trains far more stably at small scale than
            # the original post-LN block, which usually needs a warmup.
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor is a fast path for padded, variable-length
        # sequences. Every row here has exactly n_tokens tokens and none are
        # padding, and torch disables it anyway under norm_first; saying so
        # explicitly keeps the warning out of the training logs.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialise the learned tokens with small random values.

        Zero-initialised tokens are symmetric and give the encoder nothing to
        differentiate, so both are drawn from a narrow normal instead.
        """
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.feature_embedding, std=0.02)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration sufficient to rebuild this model."""
        return {
            "type": "attn",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Map a batch to survival logits.

        Args:
            x_num: Float tensor ``(B, n_numeric)``.
            x_cat: Long tensor ``(B, n_categorical)`` of vocabulary indices.

        Returns:
            Float tensor ``(B,)`` of logits.
        """
        batch_size = x_num.shape[0]

        tokens: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]
        # unsqueeze(-1) makes each scalar a (B, 1) input for Linear(1, d), and
        # unsqueeze(1) then turns the (B, d) result into a one-token sequence.
        tokens.extend(
            projection(x_num[:, i : i + 1]).unsqueeze(1)
            for i, projection in enumerate(self.num_projections)
        )
        tokens.extend(
            embedding(x_cat[:, i]).unsqueeze(1) for i, embedding in enumerate(self.cat_embeddings)
        )

        sequence = torch.cat(tokens, dim=1) + self.feature_embedding
        encoded = self.encoder(sequence)
        # Token 0 is [CLS]: after attention it has aggregated the whole row.
        return self.head(encoded[:, 0]).squeeze(-1)


#: Registry of the torch architectures, keyed by the name used everywhere else
#: (the CLI flag, the artifact directory, the registry entry, the app radio).
MODEL_TYPES = ("fast", "deep", "attn")


def build_model(config: dict[str, Any], n_numeric: int, cardinalities: list[int]) -> nn.Module:
    """Construct a model from its serialised configuration.

    This is the function ``artifacts.load_bundle`` uses to rebuild a model
    before loading its ``state_dict``, so it must accept exactly what the
    model's own ``config`` property produces.

    Args:
        config: Must contain ``"type"``; other keys are architecture
            hyperparameters and fall back to the class defaults.
        n_numeric: Number of numeric features from the preprocessor.
        cardinalities: Vocabulary sizes from the preprocessor.

    Returns:
        An initialised, untrained model.

    Raises:
        ValueError: If ``type`` is missing or unrecognised, listing the valid
            options.
    """
    model_type = config.get("type")
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model type {model_type!r}. Valid options: {', '.join(MODEL_TYPES)}."
        )

    if model_type == "fast":
        return TitanicLinear(n_numeric, cardinalities)

    if model_type == "deep":
        return TitanicMLP(
            n_numeric,
            cardinalities,
            hidden=config.get("hidden", (64, 32)),
            dropout=config.get("dropout", 0.3),
        )

    return TitanicAttention(
        n_numeric,
        cardinalities,
        d_model=config.get("d_model", 16),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 64),
        dropout=config.get("dropout", 0.2),
    )
