# Functional libraries
import numpy as np
from spacy.tokens import Doc, Token

# Utilities
from multiset import Multiset
from itertools import product

# Type hints
from abc import ABC, abstractmethod
from typing import Tuple, List, TypedDict


class ExampleType(TypedDict):
    lemma: str
    lemmas: List[str]
    example: Doc
    match_idx: Tuple[int, ...]
    pos: str


class Filter(ABC):
    @staticmethod
    @abstractmethod
    def default_data() -> any:
        raise NotImplementedError

    @abstractmethod
    def filter(self, data: any, sent: Doc, match_idx: Tuple[int, ...]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add_example(self, data: any, mwe: ExampleType) -> None:
        raise NotImplementedError


class F1Data:
    List


class F1(Filter):
    """F1: Components should be disambiguated
    # This filter keeps track of all the multisets of POS tags, observed in the training set. It only accepts candidate matches whose POS multiset was observed during training.
    # This filter runs the risk of degrading in significance when example sentences are automatically POS-tagged. Every wrong POS tag will reduce the filter's strictness and, thus, increase recall of the whole system. The same holds true for fiters F2 and F3.
    """

    def _get_multiset(self, doc: Doc, match_idx: Tuple[int, ...]):
        return sorted([doc[i].pos_ for i in match_idx])

    def add_example(self, pos_sets: F1Data, mwe: ExampleType):
        pos_multiset = self._get_multiset(mwe["example"], mwe["match_idx"])
        if pos_multiset not in pos_sets:
            pos_sets.append(pos_multiset)

    def filter(self, pos_sets: F1Data, sent: Doc, match_idx: Tuple[int, ...]):
        match_pos = self._get_multiset(sent, match_idx)
        return any([Multiset(match_pos) <= Multiset(pos_set) for pos_set in pos_sets])

    def default_data() -> F1Data:
        return []


class F2Data:
    List


class F2(Filter):
    """
    F2: Components should appear in specific orders *disregarding discontinuities*
    This filter accepts only candidate matches whose order of POS-tags has been observed in the training set.
    """

    def _get_pos(self, doc: Doc, match_idx: Tuple[int, ...]):
        return [doc[i].pos_ for i in match_idx]

    def add_example(self, pos_orders: F2Data, mwe: ExampleType):
        pos_order = self._get_pos(mwe["example"], mwe["match_idx"])
        if pos_order not in pos_orders:
            pos_orders.append(pos_order)

    def filter(self, pos_orders: F2Data, sent: Doc, match_idx: Tuple[int, ...]):
        match_pos = self._get_pos(sent, match_idx)
        return any([match_pos == pos_order for pos_order in pos_orders])

    def default_data() -> F2Data:
        return []


class F3Data:
    List


class F3(Filter):
    """
    F3: Components should appear in specific orders *considering discontinuities*
    The same as F2 but considering all POS tags from the first candidate token to the last candidate token (considering discontinuities).
    """

    def _get_pos(self, doc: Doc, match_idx: Tuple[int, ...]):
        return [doc[i].pos_ for i in range(min(match_idx), max(match_idx) + 1)]

    def add_example(self, pos_orders: F3Data, mwe: ExampleType):
        pos_order = self._get_pos(mwe["example"], mwe["match_idx"])
        if pos_order not in pos_orders:
            pos_orders.append(pos_order)

    def filter(self, pos_orders: F3Data, sent: Doc, match_idx: Tuple[int, ...]):
        match_pos = self._get_pos(sent, match_idx)
        return any([match_pos == pos_order for pos_order in pos_orders])

    def default_data() -> F3Data:
        return []


class F4Data:
    List


class F4(Filter):
    """
    F4: Components should not be too far
    This filter only accepts candidate matches where the largest discontinuity is at most the largest discontinuity, observed in the training set.
    """

    def __init__(self):
        self.discontinuities = []

    def _get_discontinuity(self, match_idx: Tuple[int, ...]):
        idx = list(match_idx)
        idx.sort()
        return max([idx[i + 1] - idx[i] for i in range(len(idx) - 1)])

    def add_example(self, discontinuities: F4Data, mwe: ExampleType):
        discontinuity = self._get_discontinuity(mwe["match_idx"])
        if discontinuity not in discontinuities:
            discontinuities.append(discontinuity)

    def filter(self, discontinuities: F4Data, sent: Doc, match_idx: Tuple[int, ...]):
        match_discontinuity = self._get_discontinuity(match_idx)
        return match_discontinuity <= max(discontinuities)

    def default_data() -> F4Data:
        return []


class F5Data:
    None


class F5(Filter):
    """
    F5: Closer components are preferred over distant ones
    This filter is global (i.e. works in the same way for all MWEs). It lets a candidate match pass only if this match has the smallest discontinuity for all other matches of the given multiset of lemmata in the given sentence.
    """

    def _get_discontinuity(self, doc: Doc, match_idx: Tuple[int, ...]):
        idx = list(match_idx)
        idx.sort()
        return max([idx[i + 1] - idx[i] for i in range(len(idx) - 1)])

    def _get_all_internal_matches(self, doc: Doc, match_idx: Tuple[int, ...]):
        lemmas = [doc[i].lemma_ for i in match_idx]
        single_matches = []

        for lemma in lemmas:
            single_matches.append(
                [
                    tok.i
                    for tok in list(
                        filter(lambda x: x.lemma_.lower() == lemma.lower(), doc)
                    )
                ]
            )
        return list(product(*single_matches))

    def filter(self, data: F5Data, sent: Doc, match_idx: Tuple[int, ...]):
        all_matches = self._get_all_internal_matches(sent, match_idx)
        return self._get_discontinuity(sent, match_idx) <= min(
            [self._get_discontinuity(sent, other_match) for other_match in all_matches]
        )

    def add_example(self, data: F5Data, mwe: ExampleType):
        return None

    def default_data() -> F5Data:
        return None


class F6Data:
    None


class F6(Filter):
    """
    F6: Components should be syntactically connected
    This filter is global. It keeps a candidate match of two tokens if the tokens are parents or grandparents of each other. It keeps candidate matches of more than two tokens if these tokens build a connected subgraph of the dependency tree.
    """

    def _is_connected_tree(self, tokens: List[Token]):
        # Algorithm has high complexity,
        # Usable only since number of tokens is not expected to be large
        N = len(tokens)
        adjacency = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if i == j:
                    adjacency[i][j] = 1
                    continue
                adjacency[i][j] = (
                    1
                    if tokens[i].head == tokens[j] or tokens[j].head == tokens[i]
                    else 0
                )
        power = np.linalg.matrix_power(adjacency, N)
        return any([all(row > 0) for row in power])

    def filter(self, data: F6Data, sent: Doc, match_idx: Tuple[int, ...]):
        if len(match_idx) == 2:
            a, b = sent[match_idx[0]], sent[match_idx[1]]
            return a.head == b or b.head == a or a.head.head == b or b.head.head == a

        return self._is_connected_tree([sent[i] for i in match_idx])

    def add_example(self, data: F6Data, mwe: ExampleType):
        return None

    def default_data() -> F6Data:
        return None


class F7Data:
    List


class F7(Filter):
    """ "
    F7: Nominal components should appear with a seen inflection
    If a candidate match has exactly one noun, this filter expects this noun to have a previous observed inflection. If a candidate match has no or more than one noun, it automatically passes this filter.
    """

    def _get_nouns(self, doc: Doc, match_idx: Tuple[int, ...]):
        return list(filter(lambda x: x.pos_ == "NOUN", [doc[i] for i in match_idx]))

    def _get_noun_morphs(self, doc: Doc, match_idx: Tuple[int, ...]):
        nouns = self._get_nouns(doc, match_idx)
        return [tok.morph.to_json() for tok in nouns]

    def add_example(self, noun_morphs: F7Data, mwe: ExampleType):
        noun_morph = self._get_noun_morphs(mwe["example"], mwe["match_idx"])
        if noun_morph not in noun_morphs:
            noun_morphs.append(noun_morph)

    def filter(self, noun_morphs: F7Data, sent: Doc, match_idx: Tuple[int, ...]):
        nouns = self._get_nouns(sent, match_idx)
        if not len(nouns) == 1:
            return True
        noun = nouns[0]
        return noun.morph.to_json() in noun_morphs

    def default_data() -> F7Data:
        return []


class F8Data:
    None


class F8(Filter):
    """
    F8: Nested VMWEs should be annotated as in train
    This filter is global. It lets a candidate match pass only if this match is not nested in another match of the same multiset of lemmata in the given sentence.
    """

    def filter(self, data: F8Data, sent: Doc, match_idx: Tuple[int, ...]):
        return True

    def add_example(self, data: F8Data, mwe: ExampleType):
        return None

    def default_data() -> F8Data:
        return None
