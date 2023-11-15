from collections import defaultdict
from itertools import combinations, product
from typing import DefaultDict, Iterable, List, Protocol, Tuple, Union

import numpy as np


def checkConsecutive(l: Tuple[int, ...]):
    n = len(l) - 1
    return sum(np.diff(sorted(l)) == 1) >= n


def find_candidate_matches(
    lemmas: List[str], token_lemmas: List[str]
) -> List[Tuple[int, ...]]:
    lemma_counts: DefaultDict[str, int] = defaultdict(int)
    for lemma in lemmas:
        lemma_counts[lemma.lower()] += 1

    drawn: List[Union[List[int], List[Tuple[int, ...]]]] = []
    for lemma, count in lemma_counts.items():
        matched_tokens = [
            i for i, tok_lemma in enumerate(token_lemmas) if tok_lemma.lower() == lemma
        ]
        if len(matched_tokens) == 0:
            return []

        if count > 1:
            comb = list(combinations(matched_tokens, count))
            drawn.append(comb)
        else:
            drawn.append(matched_tokens)

    candidate_matches = product(*drawn) if len(drawn) > 0 else []

    match_idxs: List[Tuple[int, ...]] = []
    for tuple_item in candidate_matches:
        flattened_tuple: List[int] = []
        for item in tuple_item:
            if isinstance(item, tuple):
                flattened_tuple.extend(item)
            else:
                flattened_tuple.append(item)
        match_idxs.append(tuple(sorted(flattened_tuple)))

    return match_idxs


def find_continuous_candidate_matches(
    lemmas: List[str], token_lemmas: List[str]
) -> List[Tuple[int, ...]]:
    match_idxs_with_discontinuities = find_candidate_matches(lemmas, token_lemmas)
    match_idxs: List[Tuple[int, ...]] = []
    for match_idx in match_idxs_with_discontinuities:
        if len(match_idx) == 1:
            match_idxs.append(match_idx)
            continue

        if checkConsecutive(match_idx):
            match_idxs.append(tuple(match_idx))
    return match_idxs
