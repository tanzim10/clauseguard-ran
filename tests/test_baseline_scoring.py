import pytest

from clauseguard.baselines.contracts import CasePrediction
from clauseguard.baselines.scoring import score_combined_diagnosis, score_predictions


def prediction(scenario_id, task, label=None, status="ok", error=None):
    return CasePrediction(scenario_id=scenario_id, task=task, predicted_label=label,
                          status=status, error=error)


def test_detection_metrics_count_misses_false_alarms_and_failed_outputs():
    expected = {"normal-ok": 0, "normal-fp": 0, "fault-ok": 1, "fault-miss": 1,
                "fault-error": 1, "fault-invalid": 1}
    predictions = [prediction("normal-ok", "detection", 0),
                   prediction("normal-fp", "detection", 1),
                   prediction("fault-ok", "detection", 1),
                   prediction("fault-miss", "detection", 0),
                   prediction("fault-error", "detection", status="error", error="timeout"),
                   prediction("fault-invalid", "detection", 9)]

    result = score_predictions("detection", expected, {sid: sid for sid in expected}, predictions,
                               bootstrap_replicates=0)

    assert result["count"] == 6
    assert result["accuracy"] == pytest.approx(2 / 6)
    assert result["precision"] == pytest.approx(1 / 2)
    assert result["recall"] == pytest.approx(1 / 4)
    assert result["balanced_accuracy"] == pytest.approx(3 / 8)
    assert result["normal_false_positive_rate"] == pytest.approx(1 / 2)
    assert result["missed_faults"] == 1
    assert result["prediction_status_counts"] == {"ok": 4, "error": 1, "invalid": 1, "missing": 0}
    assert result["successful_only"]["accuracy"] == pytest.approx(1 / 2)
    assert result["confusion_matrix"]["1"]["missing"] == 2


def test_missing_prediction_and_duplicate_or_out_of_cohort_ids():
    result = score_predictions("category", {"a": 2}, {"a": "g1"}, [], bootstrap_replicates=0)
    assert result["accuracy"] == 0
    assert result["prediction_status_counts"]["missing"] == 1
    assert result["confusion_matrix"]["2"]["missing"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions("category", {"a": 2}, {"a": "g1"},
                          [prediction("a", "category", 2), prediction("a", "category", 2)])
    with pytest.raises(ValueError, match="outside cohort"):
        score_predictions("category", {"a": 2}, {"a": "g1"}, [prediction("b", "category", 2)])


def test_category_wrong_class_and_support_are_hand_calculable():
    expected = {"a": 0, "b": 1}
    result = score_predictions("category", expected, {"a": "g1", "b": "g2"},
                               [prediction("a", "category", 1), prediction("b", "category", 1)],
                               bootstrap_replicates=0)
    assert result["accuracy"] == pytest.approx(.5)
    assert result["per_class"]["0"] == {"support": 1, "precision": 0, "recall": 0, "f1": 0}
    assert result["per_class"]["1"]["support"] == 1
    assert result["confusion_matrix"]["0"]["1"] == 1


def test_combined_diagnosis_composes_detection_and_category_heads():
    det_targets = {"normal": 0, "fault": 1, "jam": 1}
    cat_targets = {"fault": 3}
    groups = {"normal": "g0", "fault": "g1"}
    result = score_combined_diagnosis(
        det_targets, cat_targets, groups,
        [prediction("normal", "detection", 0), prediction("fault", "detection", 1),
         prediction("jam", "detection", 1)],
        [prediction("fault", "category", 3)], bootstrap_replicates=0,
    )
    assert result["count"] == 2
    assert result["accuracy"] == 1
    assert result["class_support"]["0"] == 1
    assert result["class_support"]["4"] == 1


def test_group_bootstrap_is_reproducible_and_requires_ten_groups():
    expected = {f"case-{i}": i % 2 for i in range(10)}
    groups = {sid: f"recording-{i}" for i, sid in enumerate(expected)}
    predictions = [prediction(sid, "detection", target) for sid, target in expected.items()]
    first = score_predictions("detection", expected, groups, predictions, bootstrap_replicates=100)
    second = score_predictions("detection", expected, groups, predictions, bootstrap_replicates=100)
    assert first["confidence_intervals"] == second["confidence_intervals"]
    assert first["confidence_intervals"]["available"]
    few = score_predictions("detection", {"a": 0}, {"a": "one"}, [], bootstrap_replicates=5)
    assert not few["confidence_intervals"]["available"]
    assert "10 recording groups" in few["confidence_intervals"]["reason"]
