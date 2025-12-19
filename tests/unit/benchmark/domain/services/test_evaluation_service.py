"""
Unit tests for EvaluationService.

These tests verify pure evaluation logic with no I/O dependencies.
All tests are deterministic and should run quickly.
"""

import pytest
from benchmark.domain.services import EvaluationService
from benchmark.domain.value_objects import CorrectnessResult


class TestEvaluateCorrectness:
    """Tests for evaluate_correctness() method."""

    def test_evaluate_correctness_perfect_match(self):
        """All expected entities are found in response."""
        response = "The Quarkonium particle was discovered during the Chronokinetics experiment."
        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.5
        )

        assert isinstance(result, CorrectnessResult)
        assert result.is_correct is True
        assert result.correctness_score == 1.0
        assert result.matched_entities == ["Quarkonium", "Chronokinetics"]
        assert result.expected_entities == expected_entities
        assert "2/2" in result.explanation

    def test_evaluate_correctness_partial_match_above_threshold(self):
        """Some entities found, above recall threshold."""
        response = "The Quarkonium particle was observed in recent studies."
        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.5
        )

        assert result.is_correct is True
        assert result.correctness_score == 0.5
        assert result.matched_entities == ["Quarkonium"]
        assert "1/2" in result.explanation

    def test_evaluate_correctness_partial_match_below_threshold(self):
        """Some entities found, but below recall threshold."""
        response = "The Quarkonium particle was observed in recent studies."
        expected_entities = ["Quarkonium", "Chronokinetics", "Temporalysis"]

        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.5
        )

        assert result.is_correct is False
        assert result.correctness_score == pytest.approx(0.333, abs=0.01)
        assert result.matched_entities == ["Quarkonium"]
        assert "1/3" in result.explanation

    def test_evaluate_correctness_no_match(self):
        """No entities found in response."""
        response = "This is a completely unrelated response about other topics."
        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.5
        )

        assert result.is_correct is False
        assert result.correctness_score == 0.0
        assert result.matched_entities == []
        assert "0/2" in result.explanation

    def test_evaluate_correctness_empty_response(self):
        """Response is empty or too short."""
        result = EvaluationService.evaluate_correctness(
            response="",
            expected_entities=["Quarkonium"],
            recall_threshold=0.5
        )

        assert result.is_correct is False
        assert result.correctness_score == 0.0
        assert "too short" in result.explanation.lower()

    def test_evaluate_correctness_response_too_short(self):
        """Response doesn't meet minimum length."""
        result = EvaluationService.evaluate_correctness(
            response="Short",
            expected_entities=["Quarkonium"],
            recall_threshold=0.5,
            min_response_length=20
        )

        assert result.is_correct is False
        assert result.correctness_score == 0.0

    def test_evaluate_correctness_no_expected_entities(self):
        """No entities to evaluate - edge case."""
        response = "This is a valid response."
        expected_entities = []

        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.5
        )

        assert result.is_correct is True
        assert result.correctness_score == 0.5
        assert "No entities to evaluate" in result.explanation

    def test_evaluate_correctness_custom_threshold(self):
        """Custom recall threshold."""
        response = "The Quarkonium particle was observed."
        expected_entities = ["Quarkonium", "Chronokinetics", "Temporalysis"]

        # With 0.3 threshold, 1/3 should pass
        result = EvaluationService.evaluate_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.3
        )

        assert result.is_correct is True
        assert result.correctness_score == pytest.approx(0.333, abs=0.01)


class TestEvaluateSuperpositionCorrectness:
    """Tests for evaluate_superposition_correctness() method."""

    def test_evaluate_superposition_correctness_single_option_correct(self):
        """Response with single option containing all entities."""
        response = "The Quarkonium particle was discovered during Chronokinetics experiments."
        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_superposition_correctness(
            response=response,
            expected_entities=expected_entities
        )

        assert result.is_correct is True
        assert result.correctness_score == 1.0
        assert result.matched_entities == ["Quarkonium", "Chronokinetics"]

    def test_evaluate_superposition_correctness_multiple_options_one_correct(self):
        """Response with multiple options, one contains correct answer."""
        response = """There are two perspectives:
- The Quarkonium particle was discovered during Chronokinetics experiments.
- Alternative theory suggests Neutrinium is more relevant."""

        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_superposition_correctness(
            response=response,
            expected_entities=expected_entities
        )

        assert result.is_correct is True
        assert result.correctness_score == 1.0
        assert set(result.matched_entities) == {"Quarkonium", "Chronokinetics"}
        assert "Found correct answer" in result.explanation

    def test_evaluate_superposition_correctness_multiple_options_none_correct(self):
        """Response with multiple options, none meet threshold."""
        response = """There are two perspectives:
- The Neutrinium particle was discovered.
- Alternative theory suggests Positronium."""

        expected_entities = ["Quarkonium", "Chronokinetics"]

        result = EvaluationService.evaluate_superposition_correctness(
            response=response,
            expected_entities=expected_entities
        )

        assert result.is_correct is False
        assert result.correctness_score == 0.0
        assert "No option had sufficient entities" in result.explanation

    def test_evaluate_superposition_correctness_partial_in_multiple_options(self):
        """Each option has partial match, best one selected."""
        response = """Perspectives:
- The Quarkonium particle was discovered.
- Chronokinetics experiments showed results."""

        expected_entities = ["Quarkonium", "Chronokinetics", "Temporalysis"]

        result = EvaluationService.evaluate_superposition_correctness(
            response=response,
            expected_entities=expected_entities,
            recall_threshold=0.3
        )

        # Best option has 1/3, which meets 0.3 threshold
        assert result.is_correct is True
        assert result.correctness_score == pytest.approx(0.333, abs=0.01)

    def test_evaluate_superposition_correctness_empty_response(self):
        """Empty response."""
        result = EvaluationService.evaluate_superposition_correctness(
            response="",
            expected_entities=["Quarkonium"]
        )

        assert result.is_correct is False
        assert result.correctness_score == 0.0

    def test_evaluate_superposition_correctness_no_entities(self):
        """No entities to check."""
        response = "This is a valid response with multiple perspectives."

        result = EvaluationService.evaluate_superposition_correctness(
            response=response,
            expected_entities=[]
        )

        assert result.is_correct is True
        assert result.correctness_score == 0.5


class TestEvaluateSuperpositionAwareness:
    """Tests for evaluate_superposition_awareness() method."""

    def test_evaluate_superposition_awareness_detected_bullet_list(self):
        """Response with bullet-point options detected."""
        response = """Two perspectives exist:
- The Quarkonium particle was discovered.
- Alternative view suggests different findings."""

        result = EvaluationService.evaluate_superposition_awareness(
            response=response,
            min_options=2
        )

        assert result["has_multiple_perspectives"] is True
        assert result["option_count"] >= 2
        assert len(result["options"]) >= 2
        assert "distinct options" in result["explanation"]

    def test_evaluate_superposition_awareness_detected_numbered_list(self):
        """Response with numbered list detected."""
        response = """1. The Quarkonium particle was discovered in 2015.
2. Alternative theory from 2018 suggests different mechanism."""

        result = EvaluationService.evaluate_superposition_awareness(
            response=response,
            min_options=2
        )

        assert result["has_multiple_perspectives"] is True
        assert result["option_count"] >= 2

    def test_evaluate_superposition_awareness_not_detected(self):
        """Response with single perspective."""
        response = "The Quarkonium particle was discovered in 2015."

        result = EvaluationService.evaluate_superposition_awareness(
            response=response,
            min_options=2
        )

        assert result["has_multiple_perspectives"] is False
        assert result["option_count"] < 2
        assert "single view" in result["explanation"]

    def test_evaluate_superposition_awareness_custom_min_options(self):
        """Custom minimum options threshold."""
        response = """Multiple views:
- First option with enough length
- Second option with enough length
- Third option with enough length"""

        # Require at least 3 options
        result = EvaluationService.evaluate_superposition_awareness(
            response=response,
            min_options=3
        )

        assert result["has_multiple_perspectives"] is True
        assert result["option_count"] >= 3

    def test_evaluate_superposition_awareness_filters_short_options(self):
        """Short noise options are filtered out."""
        response = """Options:
- Hi
- The Quarkonium particle was discovered with significant evidence.
- OK"""

        result = EvaluationService.evaluate_superposition_awareness(
            response=response,
            min_options=2,
            min_option_length=15
        )

        # Only one option meets minimum length
        assert result["has_multiple_perspectives"] is False


class TestExtractOptions:
    """Tests for _extract_options() helper method."""

    def test_extract_options_bullet_list(self):
        """Parse bullet-point list."""
        response = """- First option with sufficient length
- Second option also long enough
- Third option here"""

        options = EvaluationService._extract_options(response, min_length=10)

        assert len(options) >= 2
        assert any("First option" in opt for opt in options)
        assert any("Second option" in opt for opt in options)

    def test_extract_options_numbered_list(self):
        """Parse numbered list."""
        response = """1. First option with sufficient length
2. Second option also long enough"""

        options = EvaluationService._extract_options(response, min_length=10)

        assert len(options) >= 2

    def test_extract_options_plain_text(self):
        """Plain text without structure returns single option."""
        response = "This is a plain text response without any list structure."

        options = EvaluationService._extract_options(response, min_length=10)

        assert len(options) == 1
        assert options[0] == response

    def test_extract_options_empty_response(self):
        """Empty response returns empty list."""
        options = EvaluationService._extract_options("", min_length=10)

        assert len(options) == 0

    def test_extract_options_filters_by_length(self):
        """Options below minimum length are filtered."""
        response = """- Long option that meets minimum length requirement
- Short
- Another long option that also meets requirement"""

        options = EvaluationService._extract_options(response, min_length=20)

        assert len(options) >= 2
        assert all(len(opt) >= 20 for opt in options)

    def test_extract_options_asterisk_bullets(self):
        """Parse asterisk-style bullets."""
        response = """* First option with sufficient length
* Second option also long enough"""

        options = EvaluationService._extract_options(response, min_length=10)

        assert len(options) >= 2


class TestExtractEntitiesFromQuestion:
    """Tests for extract_entities_from_question() method."""

    def test_extract_entities_from_question_basic(self):
        """Extract capitalized entities from question."""
        question = "What was the role of Quarkonium in the Chronokinetics experiment?"

        entities = EvaluationService.extract_entities_from_question(question)

        assert "Quarkonium" in entities
        assert "Chronokinetics" in entities
        assert len(entities) == 2

    def test_extract_entities_from_question_filters_common_words(self):
        """Common words are filtered out."""
        question = "What is the relationship between Quarkonium and The experiment?"

        entities = EvaluationService.extract_entities_from_question(question)

        assert "Quarkonium" in entities
        assert "What" not in entities
        assert "The" not in entities

    def test_extract_entities_from_question_removes_punctuation(self):
        """Punctuation is stripped from entities."""
        question = "Was Quarkonium, discovered during Chronokinetics?"

        entities = EvaluationService.extract_entities_from_question(question)

        assert "Quarkonium" in entities
        assert "Chronokinetics" in entities

    def test_extract_entities_from_question_empty(self):
        """Empty question returns empty list."""
        entities = EvaluationService.extract_entities_from_question("")

        assert entities == []

    def test_extract_entities_from_question_no_entities(self):
        """Question with no capitalized terms."""
        question = "what happened during the experiment?"

        entities = EvaluationService.extract_entities_from_question(question)

        assert len(entities) == 0

    def test_extract_entities_from_question_multiple_same_entity(self):
        """Same entity appears multiple times."""
        question = "Did Quarkonium interact with Quarkonium particles?"

        entities = EvaluationService.extract_entities_from_question(question)

        # Should include both occurrences
        assert entities.count("Quarkonium") == 2


class TestCorrectnessResultValidation:
    """Tests for CorrectnessResult value object validation."""

    def test_correctness_result_valid_score(self):
        """Valid correctness score."""
        result = CorrectnessResult(
            is_correct=True,
            correctness_score=0.75,
            matched_entities=["A", "B"],
            expected_entities=["A", "B", "C"],
            explanation="Test"
        )

        assert result.correctness_score == 0.75

    def test_correctness_result_invalid_score_too_high(self):
        """Score above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="correctness_score must be in"):
            CorrectnessResult(
                is_correct=True,
                correctness_score=1.5,
                matched_entities=["A"],
                expected_entities=["A"],
                explanation="Test"
            )

    def test_correctness_result_invalid_score_too_low(self):
        """Score below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="correctness_score must be in"):
            CorrectnessResult(
                is_correct=False,
                correctness_score=-0.1,
                matched_entities=[],
                expected_entities=["A"],
                explanation="Test"
            )

    def test_correctness_result_boundary_values(self):
        """Boundary values 0.0 and 1.0 are valid."""
        result_zero = CorrectnessResult(
            is_correct=False,
            correctness_score=0.0,
            matched_entities=[],
            expected_entities=["A"],
            explanation="Test"
        )
        assert result_zero.correctness_score == 0.0

        result_one = CorrectnessResult(
            is_correct=True,
            correctness_score=1.0,
            matched_entities=["A"],
            expected_entities=["A"],
            explanation="Test"
        )
        assert result_one.correctness_score == 1.0
