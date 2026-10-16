import pytest

from clauseguard.datasets.contracts import LABEL_MAPPINGS, RecordingReview


def test_recording_review_is_separate_from_raw_match():
    assert RecordingReview().status == "unresolved"
    with pytest.raises(ValueError, match="verified recording review"):
        RecordingReview(status="verified", recording_id="session-1")
    review = RecordingReview(status="verified", recording_id="session-1", evidence="Session ledger",
                             reviewer="human", reviewed_at="2026-10-15")
    assert review.status == "verified"
    assert len(LABEL_MAPPINGS["category"]) == 10
    assert LABEL_MAPPINGS["detection"] == {"normal": 0, "abnormal": 1}
