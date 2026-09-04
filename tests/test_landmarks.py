"""Tests for the landmark definitions.

Cheap, but they catch a genuinely nasty class of bug: a mistyped landmark
index produces a skeleton that looks almost right, and a squat detector that
silently measures the wrong joint.
"""

from kinetirun.vision import POSE_CONNECTIONS, Landmark
from kinetirun.vision.pose_estimator import PoseEstimationError, PoseEstimator

import pytest


def test_blazepose_has_33_landmarks():
    assert len(Landmark) == 33


def test_indices_are_unique_and_contiguous():
    values = sorted(int(member) for member in Landmark)
    assert values == list(range(33))


def test_connections_reference_real_landmarks():
    for start, end in POSE_CONNECTIONS:
        assert 0 <= int(start) < 33
        assert 0 <= int(end) < 33
        assert start != end


def test_no_duplicate_connections():
    """A duplicate would be drawn twice - harmless, but a sign of a copy-paste
    slip in a table that is easy to get wrong."""
    normalized = {frozenset((int(a), int(b))) for a, b in POSE_CONNECTIONS}
    assert len(normalized) == len(POSE_CONNECTIONS)


def test_missing_model_gives_actionable_error(tmp_path):
    """The failure mode a new clone hits first: no .task file downloaded yet."""
    estimator = PoseEstimator(model_path=tmp_path / "does_not_exist.task")

    with pytest.raises(PoseEstimationError, match="download_model"):
        estimator.open()
