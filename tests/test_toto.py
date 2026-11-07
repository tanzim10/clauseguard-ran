import numpy as np
import pytest

from clauseguard.kpi.toto import (
    CHANNELS,
    CONTEXT,
    HORIZON,
    QUANTILES,
    apply_threshold,
    fill_context,
    fit_scales,
    forecast_median,
    last_value_forecast,
    select_threshold,
    split_windows,
    window_scores,
)

torch = pytest.importorskip("torch")


class FakeToto(torch.nn.Module):
    """Median = last context value; records every call so tests can inspect model inputs."""

    def __init__(self, shape=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_head = type("Head", (), {"knots": list(QUANTILES)})()
        self.calls, self.shape = [], shape

    def forecast(self, inputs, *, horizon, decode_block_size, has_missing_values):
        self.calls.append({**inputs, "horizon": horizon, "decode_block_size": decode_block_size,
                           "has_missing_values": has_missing_values})
        last = inputs["target"][..., -1:].repeat(1, 1, horizon)
        result = torch.stack([last + (q - 0.5) for q in QUANTILES])
        return result if self.shape is None else torch.zeros(self.shape)


def _windows(batch=3):
    values = np.arange(batch * CHANNELS * 128, dtype=np.float64).reshape(batch, CHANNELS, 128)
    return values


def test_split_keeps_the_future_out_of_the_context():
    windows = _windows()
    windows[0, 2, 100] = np.nan
    context, mask, target, target_mask = split_windows(windows)
    assert context.shape == (3, CHANNELS, CONTEXT) and target.shape == (3, CHANNELS, HORIZON)
    assert np.array_equal(context, windows[..., :96]) and np.array_equal(target[1], windows[1, :, 96:])
    assert mask.all() and not target_mask[0, 2, 4] and target_mask.sum() == 3 * CHANNELS * 32 - 1
    with pytest.raises(ValueError, match="expected windows"):
        split_windows(windows[..., :100])


def test_scales_use_observed_normal_training_values_only():
    windows = np.zeros((3, CHANNELS, 128))
    windows[0, 0] = np.r_[np.zeros(64), np.full(64, 2.0)]  # normal: SD 1
    windows[1, 0] = 1000.0 * np.arange(128)                 # fault: ignored
    windows[0, 1, :10] = np.nan
    scales = fit_scales(windows, [0, 1, 0])
    assert scales[0] == pytest.approx(np.std(np.r_[np.zeros(64), np.full(64, 2.0), np.zeros(128)]))
    assert scales[1] == 1.0  # constant channel
    windows[[0, 2], 3] = np.nan
    with pytest.raises(ValueError, match="observed normal training"):
        fit_scales(windows, [0, 1, 0])
    with pytest.raises(ValueError, match="normal training window"):
        fit_scales(windows, [1, 1, 1])


def test_fill_and_last_value_respect_masks():
    context = np.full((1, CHANNELS, CONTEXT), 5.0)
    context[0, 0, :3] = np.nan
    context[0, 0, 10] = np.nan
    context[0, 1] = np.nan
    mask = ~np.isnan(context)
    filled = fill_context(context, mask)
    assert np.isfinite(filled).all()
    assert np.array_equal(filled[mask], context[mask])
    assert filled[0, 0, :3].tolist() == [5.0] * 3 and filled[0, 1].max() == 0.0
    context[0, 2, -1] = np.nan
    context[0, 2, -2] = 7.0
    last = last_value_forecast(context, ~np.isnan(context))
    assert last.shape == (1, CHANNELS, HORIZON)
    assert (last[0, 2] == 7.0).all() and np.isnan(last[0, 1]).all()


def test_scores_follow_the_formula_and_record_failures():
    target = np.zeros((3, CHANNELS, HORIZON))
    forecast = np.zeros_like(target)
    forecast[0, 0] = 4.0                     # MAE 4 on channel 0
    target_mask = np.ones_like(target, dtype=bool)
    target_mask[0, 0, 16:] = False           # unobserved targets are ignored
    forecast[0, 0, 16:] = 100.0
    context_mask = np.ones((3, CHANNELS, CONTEXT), dtype=bool)
    context_mask[1, 5] = False
    forecast[2, 7, 3] = np.nan
    scales = np.ones(CHANNELS)
    scales[0] = 2.0
    scores, failures = window_scores(forecast, target, target_mask, context_mask, scales)
    assert scores[0] == pytest.approx((4.0 / 2.0) / CHANNELS)
    assert np.isnan(scores[1:]).all()
    assert failures == {1: "PRB_Utilization_UL has no observed context",
                        2: "PRBs_UL_Current forecast is not finite"}
    target_mask[0, 3] = False
    assert "no observed forecast target" in window_scores(
        forecast, target, target_mask, context_mask, scales)[1][0]


def test_threshold_rule_uses_dev_macro_f1_and_documented_ties():
    scores = np.array([0.1, 0.2, 0.3, 0.9, np.nan])
    labels = [0, 0, 0, 1, 1]
    selection = select_threshold(scores, labels)
    # Every threshold in (0.3, 0.9] gives the same score and no false alarm; the highest wins.
    assert selection["threshold"] == 0.9
    assert selection["dev_macro_f1"] == pytest.approx((1 + 2 / 3) / 2)
    predicted = apply_threshold(scores, selection["threshold"])
    assert predicted.tolist() == [0, 0, 0, 1, -1]
    assert selection["candidates"] <= 102 and selection["trials"][-1]["threshold"] is None
    # Calling everything abnormal (t <= 0.5) loses to (0.5, 0.9], whose highest value wins.
    tied = select_threshold(np.array([0.5, 0.5, 0.9]), [0, 1, 1])
    assert tied["threshold"] == 0.9 and tied["dev_macro_f1"] == pytest.approx(2 / 3)
    # When nothing separates the classes, the all-normal endpoint avoids false alarms.
    flat = select_threshold(np.array([0.4, 0.4, 0.4, 0.4]), [0, 0, 0, 1])
    assert flat["threshold"] is None
    assert apply_threshold(np.array([1.0, np.nan]), None).tolist() == [0, -1]
    with pytest.raises(ValueError, match="every development forecast failed"):
        select_threshold(np.array([np.nan, np.nan]), [0, 1])


def test_forecast_uses_documented_inputs_and_checks_shape():
    windows = _windows(5)
    windows[1, 4, 7] = np.nan
    context, mask, _, _ = split_windows(windows)
    model = FakeToto()
    median = forecast_median(model, context, mask, batch_size=2)
    assert median.shape == (5, CHANNELS, HORIZON)
    assert np.allclose(median[0], np.repeat(context[0, :, -1:], HORIZON, axis=1))
    assert len(model.calls) == 3
    call = model.calls[0]
    assert call["target"].shape == (2, CHANNELS, CONTEXT) and call["horizon"] == HORIZON
    assert call["decode_block_size"] is None and call["has_missing_values"] is True
    assert call["series_ids"].shape == (2, CHANNELS) and not call["series_ids"].any()
    assert torch.isfinite(call["target"]).all() and not call["target_mask"][1, 4, 7]
    with pytest.raises(ValueError, match="Toto returned shape"):
        forecast_median(FakeToto(shape=(9, 1, CHANNELS, 16)), context, mask)
