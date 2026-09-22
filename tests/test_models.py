"""Tests for the PyTorch model ladder.

Three architectures share one interface: ``forward(x_num, x_cat) -> logits``
of shape ``(B,)``. Everything downstream -- the training loop, the artifact
format, the inference service -- depends on that being true for all of them,
so these tests check the contract rather than the internals.
"""

from __future__ import annotations

import pytest
import torch

from titanic.models import (
    TitanicAttention,
    TitanicLinear,
    TitanicMLP,
    build_model,
    count_parameters,
)

# Cardinalities of the real fitted preprocessor: Pclass, Sex, Embarked, Title,
# Deck, IsAlone -- each including the reserved <UNK> slot at index 0.
CARDINALITIES = [4, 3, 4, 6, 10, 3]
N_NUMERIC = 3
BATCH = 7


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic batch shaped like the preprocessor's output."""
    generator = torch.Generator().manual_seed(0)
    x_num = torch.randn(BATCH, N_NUMERIC, generator=generator)
    x_cat = torch.stack(
        [torch.randint(0, card, (BATCH,), generator=generator) for card in CARDINALITIES],
        dim=1,
    )
    return x_num, x_cat


ALL_MODELS = [
    pytest.param(lambda: TitanicLinear(N_NUMERIC, CARDINALITIES), id="fast"),
    pytest.param(lambda: TitanicMLP(N_NUMERIC, CARDINALITIES), id="deep"),
    pytest.param(lambda: TitanicAttention(N_NUMERIC, CARDINALITIES), id="attn"),
]


class TestForwardContract:
    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_returns_one_logit_per_row(self, build, inputs) -> None:
        x_num, x_cat = inputs
        logits = build()(x_num, x_cat)
        # (B,) not (B, 1): BCEWithLogitsLoss compares against a (B,) target,
        # and a stray trailing dimension would broadcast into a (B, B) loss.
        assert logits.shape == (BATCH,)
        assert logits.dtype == torch.float32

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_output_is_finite(self, build, inputs) -> None:
        logits = build()(*inputs)
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_handles_a_single_row(self, build) -> None:
        # Single-row inference is the app's most common request shape, and it
        # is where BatchNorm-style layers would fail. None of these models use
        # batch statistics, and this test keeps it that way.
        model = build().eval()
        x_num = torch.zeros(1, N_NUMERIC)
        x_cat = torch.zeros(1, len(CARDINALITIES), dtype=torch.long)
        assert model(x_num, x_cat).shape == (1,)

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_accepts_the_unknown_category_index(self, build) -> None:
        # Index 0 is <UNK>. Every embedding must have a row for it.
        model = build().eval()
        x_num = torch.zeros(2, N_NUMERIC)
        x_cat = torch.zeros(2, len(CARDINALITIES), dtype=torch.long)
        assert torch.isfinite(model(x_num, x_cat)).all()

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_accepts_the_highest_valid_index(self, build) -> None:
        # Off-by-one in embedding sizing would raise an index error here.
        model = build().eval()
        x_num = torch.zeros(1, N_NUMERIC)
        x_cat = torch.tensor([[card - 1 for card in CARDINALITIES]], dtype=torch.long)
        assert torch.isfinite(model(x_num, x_cat)).all()

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_gradients_reach_every_parameter(self, build, inputs) -> None:
        # A parameter with no gradient is a wiring bug: it was created but
        # never used in forward(), so training silently ignores it.
        model = build()
        model(*inputs).sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == [], f"parameters never used in forward(): {missing}"

    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_eval_mode_is_deterministic(self, build, inputs) -> None:
        # Dropout must be inactive in eval mode, or the app would return a
        # different probability for the same passenger on every click.
        model = build().eval()
        with torch.no_grad():
            first, second = model(*inputs), model(*inputs)
        torch.testing.assert_close(first, second)


class TestArchitectures:
    def test_linear_is_actually_logistic_regression(self) -> None:
        # One weight per one-hot slot plus one per numeric feature, plus bias.
        model = TitanicLinear(N_NUMERIC, CARDINALITIES)
        expected = sum(CARDINALITIES) + N_NUMERIC + 1
        assert count_parameters(model) == expected

    def test_linear_is_small_enough_to_be_interpretable(self) -> None:
        assert count_parameters(TitanicLinear(N_NUMERIC, CARDINALITIES)) < 50

    def test_mlp_embedding_dimensions_follow_the_rule_of_thumb(self) -> None:
        model = TitanicMLP(N_NUMERIC, CARDINALITIES)
        # dim = min(8, ceil(card / 2)): 4->2, 3->2, 4->2, 6->3, 10->5, 3->2
        assert [emb.embedding_dim for emb in model.embeddings] == [2, 2, 2, 3, 5, 2]

    def test_mlp_stays_small_for_712_rows(self) -> None:
        # ~3-4k parameters is about five per training row. Anything larger
        # overfits faster without adding capacity that 712 rows can support.
        n_params = count_parameters(TitanicMLP(N_NUMERIC, CARDINALITIES))
        assert 1_000 < n_params < 10_000

    def test_mlp_dropout_changes_output_in_train_mode(self, inputs) -> None:
        model = TitanicMLP(N_NUMERIC, CARDINALITIES, dropout=0.5).train()
        torch.manual_seed(0)
        first = model(*inputs)
        second = model(*inputs)
        assert not torch.allclose(first, second), "dropout appears to be inactive while training"

    def test_attention_tokenises_every_feature(self) -> None:
        model = TitanicAttention(N_NUMERIC, CARDINALITIES)
        # One token per feature plus the [CLS] token.
        assert model.n_tokens == N_NUMERIC + len(CARDINALITIES) + 1

    def test_attention_stays_within_its_parameter_budget(self) -> None:
        n_params = count_parameters(TitanicAttention(N_NUMERIC, CARDINALITIES))
        assert 2_000 < n_params < 20_000


class TestBuildModel:
    @pytest.mark.parametrize("name", ["fast", "deep", "attn"])
    def test_builds_each_registered_architecture(self, name: str) -> None:
        model = build_model({"type": name}, N_NUMERIC, CARDINALITIES)
        assert isinstance(model, torch.nn.Module)

    def test_config_hyperparameters_are_applied(self) -> None:
        model = build_model(
            {"type": "deep", "hidden": [16, 8], "dropout": 0.1}, N_NUMERIC, CARDINALITIES
        )
        assert isinstance(model, TitanicMLP)
        assert model.hidden == [16, 8]

    def test_unknown_type_names_the_valid_options(self) -> None:
        with pytest.raises(ValueError, match="fast"):
            build_model({"type": "transformer-xl"}, N_NUMERIC, CARDINALITIES)

    def test_round_trips_through_its_own_config(self) -> None:
        # save/load rebuilds a model from model_config.json, so the config a
        # model reports must be sufficient to reconstruct it exactly.
        original = build_model({"type": "deep", "hidden": [32, 16]}, N_NUMERIC, CARDINALITIES)
        rebuilt = build_model(original.config, N_NUMERIC, CARDINALITIES)
        rebuilt.load_state_dict(original.state_dict())
        assert count_parameters(rebuilt) == count_parameters(original)


class TestSeeding:
    @pytest.mark.parametrize("build", ALL_MODELS)
    def test_same_seed_gives_identical_initial_weights(self, build) -> None:
        torch.manual_seed(42)
        first = build()
        torch.manual_seed(42)
        second = build()
        for (_, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            torch.testing.assert_close(a, b)
