import pytest

from clauseguard.baselines.data import (
    ExperimentBatch,
    experiment_cohort,
    iter_experiment_batches,
)
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset


@pytest.fixture
def experiment_release(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])
from clauseguard.baselines.operations import predict_adapter, train_adapter


class FakeAdapter:
    def fit(self, batches, *, seed, configuration):
        seen = list(batches)
        return {"seen_targets": [target for batch in seen for target in batch.targets],
                "seed": seed, "configuration": dict(configuration)}

    def predict(self, model, inputs):
        assert all(not hasattr(inputs_item, "target") for inputs_item in inputs)
        return [0] * len(inputs)


def test_adapter_training_uses_train_batches_and_predict_receives_inputs_only(experiment_release):
    train = experiment_cohort(experiment_release, split="train", task="detection")
    trained_batches = list(iter_experiment_batches(experiment_release, train, batch_size=6))
    adapter = FakeAdapter()
    model = train_adapter(adapter, train, trained_batches, seed=7, configuration={"depth": 3})
    assert len(model["seen_targets"]) == 13
    assert model["seed"] == 7

    test = experiment_cohort(experiment_release, split="test", task="detection")
    batches = list(iter_experiment_batches(experiment_release, test, batch_size=8))
    predictions = predict_adapter(adapter, model, test, batches)
    assert [row.scenario_id for row in predictions] == list(test.scenario_ids)
    assert all(row.predicted_label == 0 for row in predictions)


def test_prediction_batch_failures_become_case_records(experiment_release):
    cohort = experiment_cohort(experiment_release, split="test", task="detection")
    first = cohort.scenario_ids[0]
    reduced = cohort.model_copy(update={"scenario_ids": (first,)})
    batch = ExperimentBatch(cohort_id=reduced.cohort_id, inputs=(), targets=(0,),
                            scenario_ids=(first,), group_ids=("group",))

    class FailingAdapter:
        def predict(self, model, inputs):
            raise TimeoutError("fixture")

    rows = predict_adapter(FailingAdapter(), None, reduced, [batch])
    assert rows[0].status == "error"
    assert rows[0].error == "TimeoutError: fixture"
