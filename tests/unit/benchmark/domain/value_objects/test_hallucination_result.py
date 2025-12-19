"""
Unit tests for HallucinationResult value object.
"""

import pytest
from benchmark.domain.value_objects import HallucinationResult


class TestHallucinationResultValidation:
    """Tests for HallucinationResult validation logic."""

    def test_hallucination_result_valid_no_hallucination(self):
        """Valid result with no hallucination."""
        result = HallucinationResult(
            has_hallucination=False,
            hallucination_type=None,
            severity="none",
            explanation="No issues detected",
            flagged_content=[]
        )

        assert result.has_hallucination is False
        assert result.severity == "none"

    def test_hallucination_result_valid_with_hallucination(self):
        """Valid result with hallucination detected."""
        result = HallucinationResult(
            has_hallucination=True,
            hallucination_type="invalid_term",
            severity="major",
            explanation="Found invalid terms",
            flagged_content=["FakeTerm1", "FakeTerm2"]
        )

        assert result.has_hallucination is True
        assert result.hallucination_type == "invalid_term"
        assert result.severity == "major"
        assert len(result.flagged_content) == 2

    def test_hallucination_result_invalid_severity(self):
        """Invalid severity value raises ValueError."""
        with pytest.raises(ValueError, match="severity must be one of"):
            HallucinationResult(
                has_hallucination=False,
                hallucination_type=None,
                severity="invalid",
                explanation="Test",
                flagged_content=[]
            )

    def test_hallucination_result_missing_type_when_has_hallucination(self):
        """hallucination_type must be set when has_hallucination is True."""
        with pytest.raises(ValueError, match="hallucination_type must be set"):
            HallucinationResult(
                has_hallucination=True,
                hallucination_type=None,
                severity="major",
                explanation="Test",
                flagged_content=[]
            )

    def test_hallucination_result_severity_must_be_none_when_no_hallucination(self):
        """severity must be 'none' when has_hallucination is False."""
        with pytest.raises(ValueError, match="severity must be 'none'"):
            HallucinationResult(
                has_hallucination=False,
                hallucination_type=None,
                severity="major",
                explanation="Test",
                flagged_content=[]
            )

    def test_hallucination_result_all_severity_levels(self):
        """All valid severity levels are accepted."""
        for severity in ["none", "minor", "major", "critical"]:
            if severity == "none":
                result = HallucinationResult(
                    has_hallucination=False,
                    hallucination_type=None,
                    severity=severity,
                    explanation="Test",
                    flagged_content=[]
                )
            else:
                result = HallucinationResult(
                    has_hallucination=True,
                    hallucination_type="test_type",
                    severity=severity,
                    explanation="Test",
                    flagged_content=[]
                )
            assert result.severity == severity

    def test_hallucination_result_flagged_content_can_be_empty(self):
        """flagged_content can be empty list even with hallucination."""
        result = HallucinationResult(
            has_hallucination=True,
            hallucination_type="conceptual_error",
            severity="minor",
            explanation="Conceptual issue without specific terms",
            flagged_content=[]
        )

        assert result.has_hallucination is True
        assert len(result.flagged_content) == 0
