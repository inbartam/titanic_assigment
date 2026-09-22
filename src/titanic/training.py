"""Training loop, early stopping and leak-free cross-validation.

One loop trains all three torch architectures. Sharing it is what makes the
model comparison meaningful: identical loss, optimiser, batching, seeding and
stopping rule, so any difference in validation score is attributable to the
architecture rather than to a different training recipe.

Two rules this module exists to enforce:

1. **Early stopping never sees the held-out validation split.** It monitors a
   10% stratified carve-out taken from *inside* the training split.
2. **Cross-validation refits the preprocessor inside every fold.** Fitting it
   once outside the loop would leak each fold's held-out rows into the
   imputation medians and scaling statistics.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from titanic.config import TARGET_COLUMN, TrainConfig
from titanic.models import build_model
from titanic.preprocessing import Preprocessor
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Batch size used for inference. Larger than the training batch because no
#: gradients are stored, and the whole dataset fits in memory many times over.
INFERENCE_BATCH_SIZE = 4096


class EarlyStopping:
    """Stop training when the monitored loss stops improving.

    Restores the best weights seen, so the returned model is the one from the
    best epoch rather than whatever the last epoch produced. Without that, a
    model that overfits after epoch 40 would be saved in its overfitted state
    even though training correctly stopped at epoch 60.

    Attributes:
        patience: Epochs without improvement tolerated before stopping.
        min_delta: Improvement smaller than this does not count, which stops
            the counter resetting on numerical noise.
        best_loss: Lowest loss observed so far.
        best_epoch: Epoch that produced it.
    """

    def __init__(
        self, patience: int = 20, min_delta: float = 1e-4, *, restore_best: bool = True
    ) -> None:
        """Configure the stopper.

        Args:
            patience: Epochs without improvement before stopping.
            min_delta: Minimum decrease that counts as an improvement.
            restore_best: Copy the best weights back into the model on stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.should_stop = False
        self._epochs_without_improvement = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    def step(self, loss: float, epoch: int, model: nn.Module) -> bool:
        """Record one epoch's loss and decide whether to continue.

        Args:
            loss: Monitored loss for this epoch (inner-validation loss).
            epoch: Epoch number, for reporting.
            model: The model, so the best weights can be snapshotted.

        Returns:
            ``True`` if training should stop.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self._epochs_without_improvement = 0
            if self.restore_best:
                # deepcopy onto the CPU: keeping a reference would alias the
                # live parameters and the "best" snapshot would keep changing.
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self._epochs_without_improvement += 1
            if self._epochs_without_improvement >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Load the best observed weights back into the model.

        Args:
            model: The model to restore in place.
        """
        if self.restore_best and self._best_state is not None:
            model.load_state_dict(self._best_state)


def predict_proba_torch(model: nn.Module, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
    """Run a torch model over arrays and return survival probabilities.

    Args:
        model: A trained model following the project's forward signature.
        x_num: Float array ``(n, n_numeric)``.
        x_cat: Int array ``(n, n_categorical)``.

    Returns:
        Float array ``(n,)`` of ``P(survived)`` in ``[0, 1]``.
    """
    model.eval()
    outputs: list[np.ndarray] = []

    # no_grad halves memory and skips autograd bookkeeping; eval() disables
    # dropout, without which the same passenger would score differently on
    # every request.
    with torch.no_grad():
        for start in range(0, len(x_num), INFERENCE_BATCH_SIZE):
            stop = start + INFERENCE_BATCH_SIZE
            logits = model(
                torch.from_numpy(np.asarray(x_num[start:stop], dtype=np.float32)),
                torch.from_numpy(np.asarray(x_cat[start:stop], dtype=np.int64)),
            )
            outputs.append(torch.sigmoid(logits).numpy())

    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def _make_loader(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build a seeded DataLoader over preprocessed arrays.

    Args:
        x_num: Numeric features.
        x_cat: Categorical indices.
        y: Binary targets.
        batch_size: Rows per batch.
        seed: Seed for the shuffling generator.
        shuffle: Whether to shuffle between epochs.

    Returns:
        A ``DataLoader`` yielding ``(x_num, x_cat, y)`` float/long/float
        tensors.
    """
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x_num, dtype=np.float32)),
        torch.from_numpy(np.asarray(x_cat, dtype=np.int64)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    # An explicit generator makes shuffling reproducible. num_workers=0 is
    # required on Windows: worker processes re-import the module and would
    # break inside Streamlit and uvicorn.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one pass over a loader, training if an optimiser is supplied.

    Args:
        model: The model.
        loader: Batches to iterate.
        criterion: Loss function.
        optimizer: Optimiser for training, or ``None`` to evaluate.

    Returns:
        ``(mean_loss, accuracy)`` over the pass.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    # torch.enable_grad()/no_grad() chosen by mode, so the same function serves
    # both passes without duplicating the loop.
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_num, batch_cat, batch_y in loader:
            logits = model(batch_num, batch_cat)
            loss = criterion(logits, batch_y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Weight by batch size: the final batch is usually smaller, and an
            # unweighted mean would over-count it.
            total_loss += loss.item() * len(batch_y)
            correct += ((logits > 0).float() == batch_y).sum().item()
            seen += len(batch_y)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def train_torch_model(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    config: TrainConfig | None = None,
    *,
    inner_val_size: float = 0.1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a model with early stopping on an inner carve-out.

    The carve-out is taken from the data passed in -- which is the *training
    split* -- so the held-out validation set is never involved in deciding when
    to stop.

    Args:
        model: An initialised model; trained in place.
        x_num: Numeric features of the training split.
        x_cat: Categorical indices of the training split.
        y: Binary targets.
        config: Optimiser and schedule settings. Defaults to
            :class:`~titanic.config.TrainConfig`.
        inner_val_size: Fraction of ``y`` carved out for early stopping.
        verbose: Log progress every 20 epochs.

    Returns:
        A history dict with per-epoch losses and accuracies, ``best_epoch`` and
        ``stopped_early``, ready to be written to ``history.json``.
    """
    config = config or TrainConfig()

    # Stratified carve-out: with a 38% positive rate a random 10% of 712 rows
    # can easily land at 30% or 46% positive, making the stopping signal noisy.
    indices = np.arange(len(y))
    fit_idx, inner_idx = train_test_split(
        indices, test_size=inner_val_size, random_state=config.seed, stratify=y
    )

    train_loader = _make_loader(
        x_num[fit_idx], x_cat[fit_idx], y[fit_idx], config.batch_size, config.seed, shuffle=True
    )
    inner_loader = _make_loader(
        x_num[inner_idx],
        x_cat[inner_idx],
        y[inner_idx],
        config.batch_size,
        config.seed,
        shuffle=False,
    )

    # BCEWithLogitsLoss folds the sigmoid into the loss: numerically stable at
    # extreme logits, where sigmoid-then-BCE would saturate and lose gradient.
    criterion = nn.BCEWithLogitsLoss()
    # AdamW decouples weight decay from the gradient update, which is the
    # correct L2 behaviour with adaptive optimisers.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    stopper = EarlyStopping(patience=config.patience, restore_best=True)

    history: dict[str, Any] = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, inner_loader, criterion, None)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        if verbose and (epoch % 20 == 0 or epoch == 1):
            logger.info(
                "epoch %3d | train loss %.4f acc %.3f | inner-val loss %.4f acc %.3f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

        if stopper.step(val_loss, epoch, model):
            logger.info(
                "early stop at epoch %d; restoring best weights from epoch %d (loss %.4f)",
                epoch,
                stopper.best_epoch,
                stopper.best_loss,
            )
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch
    history["best_inner_val_loss"] = round(stopper.best_loss, 6)
    history["stopped_early"] = stopper.should_stop
    history["n_epochs_run"] = len(history["epochs"])
    history["inner_val_size"] = inner_val_size
    return history


def cross_validate(
    model_config: dict[str, Any],
    df: pd.DataFrame,
    train_config: TrainConfig | None = None,
    *,
    k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Score a configuration with stratified k-fold CV inside the training split.

    The preprocessor is refitted on every fold's training portion. Fitting it
    once outside the loop is the most common subtle cross-validation bug: each
    fold's held-out rows would contribute to the imputation medians and scaling
    statistics, inflating the score.

    Args:
        model_config: Architecture config accepted by
            :func:`titanic.models.build_model`.
        df: **Engineered** training split, including the target column.
        train_config: Optimiser and schedule settings.
        k: Number of folds.
        seed: Seed for the fold split and for every model built inside it.

    Returns:
        ``{"roc_auc_mean", "roc_auc_std", "log_loss_mean", "fold_roc_auc", "k"}``.
    """
    train_config = train_config or TrainConfig()
    y = df[TARGET_COLUMN].to_numpy()
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_aucs: list[float] = []
    fold_losses: list[float] = []

    for fold, (fit_idx, score_idx) in enumerate(folds.split(df, y), start=1):
        fold_fit, fold_score = df.iloc[fit_idx], df.iloc[score_idx]

        preprocessor = Preprocessor().fit(fold_fit)
        x_num_fit, x_cat_fit = preprocessor.transform(fold_fit)
        x_num_score, x_cat_score = preprocessor.transform(fold_score)

        # Re-seed per fold so every fold starts from the same initial weights;
        # otherwise fold-to-fold variance mixes data variance with init noise.
        torch.manual_seed(seed)
        model = build_model(model_config, x_num_fit.shape[1], preprocessor.cardinalities)
        train_torch_model(model, x_num_fit, x_cat_fit, y[fit_idx], train_config, verbose=False)

        probabilities = predict_proba_torch(model, x_num_score, x_cat_score)
        fold_aucs.append(float(roc_auc_score(y[score_idx], probabilities)))
        fold_losses.append(float(log_loss(y[score_idx], probabilities, labels=[0, 1])))
        logger.debug("fold %d/%d: roc_auc=%.4f", fold, k, fold_aucs[-1])

    return {
        "k": k,
        "fold_roc_auc": [round(score, 6) for score in fold_aucs],
        "roc_auc_mean": round(float(np.mean(fold_aucs)), 6),
        "roc_auc_std": round(float(np.std(fold_aucs)), 6),
        "log_loss_mean": round(float(np.mean(fold_losses)), 6),
    }


def select_config(
    grid: list[dict[str, Any]],
    df: pd.DataFrame,
    train_config: TrainConfig | None = None,
    *,
    k: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose the best configuration in a grid by cross-validated ROC-AUC.

    Ties are broken by mean log loss, which rewards better-calibrated
    probabilities when two configurations rank cases equally well.

    The grid is deliberately tiny. On 712 rows the cross-validation standard
    deviation (~0.03 AUC) exceeds most between-configuration differences, so a
    large search would mostly be fitting noise.

    Args:
        grid: Candidate configurations.
        df: Engineered training split including the target.
        train_config: Optimiser and schedule settings.
        k: Folds per candidate.
        seed: Seed shared by every candidate, so they are compared on
            identical folds.

    Returns:
        ``(winning_config, all_results)`` where each result is the config plus
        its cross-validation scores, ready for ``history.json["cv_grid"]``.
    """
    results: list[dict[str, Any]] = []

    for position, candidate in enumerate(grid, start=1):
        scores = cross_validate(candidate, df, train_config, k=k, seed=seed)
        logger.info(
            "cv %d/%d %s -> roc_auc %.4f +/- %.4f",
            position,
            len(grid),
            {key: value for key, value in candidate.items() if key != "type"},
            scores["roc_auc_mean"],
            scores["roc_auc_std"],
        )
        results.append({"config": candidate, **scores})

    best = max(results, key=lambda r: (r["roc_auc_mean"], -r["log_loss_mean"]))
    logger.info(
        "selected %s (cv roc_auc %.4f)",
        {key: value for key, value in best["config"].items() if key != "type"},
        best["roc_auc_mean"],
    )
    return best["config"], results
