import pytest

from mwe_detector.utils import find_candidate_matches


def test_empty_input():
    assert find_candidate_matches([], []) == []


def test_no_matching_lemmas():
    lemmas = ["test1", "test2"]
    tokens = ["test3", "test4"]
    assert find_candidate_matches(lemmas, tokens) == []


def test_single_matching_lemma_single_occurrence():
    lemmas = ["test1"]
    tokens = ["test1", "test2"]
    assert find_candidate_matches(lemmas, tokens) == [(0,)]


def test_single_matching_lemma_multiple_occurrences():
    lemmas = ["test1", "test1"]
    tokens = ["test1", "test1", "test2"]
    assert find_candidate_matches(lemmas, tokens) == [(0, 1)]


def test_multiple_matching_lemmas():
    lemmas = ["test1", "test2"]
    tokens = ["test1", "test2", "test3"]
    expected_result = [(0, 1)]
    assert find_candidate_matches(lemmas, tokens) == expected_result


def test_multiple_matching_lemmas_with_redundancies():
    lemmas = ["test1", "test2"]
    tokens = ["test1", "test2", "test3", "test2"]
    expected_result = [(0, 1), (0, 3)]
    assert find_candidate_matches(lemmas, tokens) == expected_result


def test_multiple_matching_lemmas_with_duplicates():
    lemmas = ["test1", "test1", "test2", "test2"]
    tokens = [
        "test1",
        "test1",
        "test2",
        "test2",
        "test3",
    ]
    expected_result = [(0, 1, 2, 3)]
    assert find_candidate_matches(lemmas, tokens) == expected_result


def test_multiple_matching_lemmas_with_duplicates_and_redundancies():
    lemmas = ["test1", "test1", "test2", "test2"]
    tokens = [
        "test1",
        "test1",
        "test2",
        "test2",
        "test3",
        "test2",
    ]
    expected_result = [(0, 1, 2, 3), (0, 1, 2, 5), (0, 1, 3, 5)]
    assert find_candidate_matches(lemmas, tokens) == expected_result


def test_case_insensitivity():
    lemmas = ["Test1", "Test2"]
    tokens = ["test1", "test2", "test3"]
    expected_result = [(0, 1)]
    assert find_candidate_matches(lemmas, tokens) == expected_result


def test_indices_order_in_match_idxs():
    lemmas = ["test1", "test1", "test2"]
    tokens = ["test2", "test1", "test1", "test2"]
    result = find_candidate_matches(lemmas, tokens)

    # Check if each tuple in the result is sorted
    for match in result:
        assert match == tuple(sorted(match)), f"Tuple {match} is not sorted"
