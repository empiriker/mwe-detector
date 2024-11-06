# Functional libraries
import os

# Utilities
from collections import defaultdict
from pathlib import Path

# Type hints
from typing import Any, Optional, TypedDict

import srsly
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.util import ensure_path

from .filters import (
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    ExampleType,
    F1Data,
    F2Data,
    F3Data,
    F4Data,
    F5Data,
    F6Data,
    F7Data,
    F8Data,
)
from .utils import find_candidate_matches, find_continuous_candidate_matches

if not Token.has_extension("wikt_mwe"):
    Token.set_extension("wikt_mwe", default="*")


class Filters(TypedDict):
    f1: F1
    f2: F2
    f3: F3
    f4: F4
    f5: F5
    f6: F6
    f7: F7
    f8: F8


class MWEType(TypedDict):
    pos: str
    lemmas: list[str]
    f1: F1Data
    f2: F2Data
    f3: F3Data
    f4: F4Data
    f5: F5Data
    f6: F6Data
    f7: F7Data
    f8: F8Data


class MWEDetectorDataSerialized(TypedDict):
    mwes: dict[str, MWEType]
    active_filters: dict[str, list[str]]


class MWEDetectorData:
    def __init__(self):
        self.mwes: defaultdict[str, MWEType] = defaultdict(
            lambda: {
                "pos": "",
                "lemmas": [],
                "f1": F1.default_data(),
                "f2": F2.default_data(),
                "f3": F3.default_data(),
                "f4": F4.default_data(),
                "f5": F5.default_data(),
                "f6": F6.default_data(),
                "f7": F7.default_data(),
                "f8": F8.default_data(),
            }
        )
        self.active_filters: defaultdict[str, list[str]] = defaultdict(
            lambda: ["f2", "f4", "f5"],
            {
                "ADJ": ["f2", "f4", "f5"],
                "ADP": ["f2", "f4", "f5"],
                "ADV": ["f2", "f4", "f5"],
                "AUX": ["f2", "f4", "f5"],
                "CONJ": ["f2", "f4", "f5"],
                "DET": ["f2", "f4", "f5"],
                "INTJ": ["f2", "f4", "f5"],
                "NOUN": ["f2", "f4", "f5"],
                "NUM": ["f2", "f4", "f5"],
                "PART": ["f2", "f4", "f5"],
                "PRON": ["f2", "f4", "f5"],
                "PROPN": ["f2", "f4", "f5"],
                "PUNCT": ["f2", "f4", "f5"],
                "SYM": ["f2", "f4", "f5"],
                "VERB": ["f2", "f4", "f5"],
                "X": ["f2", "f4", "f5"],
            },
        )
        self.continuous_POS = ["ADJ", "ADV", "ADP", "CONJ", "INTJ", "NOUN", "PROPN"]

    def to_dict(self):
        mwes_copy = {}
        for key, value in self.mwes.items():
            value_copy = value.copy()
            value_copy["f7"] = list(value_copy["f7"])
            mwes_copy[key] = value_copy

        return {
            "mwes": mwes_copy,
            "active_filters": self.active_filters
        }

    def from_dict(self, data: MWEDetectorDataSerialized):
        self.mwes.clear()
        self.mwes.update(data["mwes"])
        self.active_filters.clear()
        self.active_filters.update(data["active_filters"])

    def __getitem__(self, key: str):
        return self.mwes[key]

    def __setitem__(self, key: str, value: MWEType):
        self.mwes[key] = value


class MWEDetector:
    def __init__(self, nlp: Language):
        self._data = MWEDetectorData()
        self._filters: Filters = {
            "f1": F1(),
            "f2": F2(),
            "f3": F3(),
            "f4": F4(),
            "f5": F5(),
            "f6": F6(),
            "f7": F7(),
            "f8": F8(),
        }
        if not nlp.lang:
            raise ValueError()
        self._lang = nlp.lang

    @property
    def mwes(self):
        return self._data.mwes

    @property
    def filters(self):
        return self._filters

    @property
    def active_filters(self):
        return self._data.active_filters

    @active_filters.setter
    def active_filters(self, new_active_filters: dict[str, list[str]]):
        self._data.active_filters = defaultdict(
            lambda: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"], new_active_filters
        )

    def _example_to_key(self, example: ExampleType):
        # lemmas = example["lemmas"]
        # lemmas.sort()
        return example["lemma"] + ":" + example["pos"]

    # def _doc_to_example_type(self, doc: Doc):
    #     match_idx = tuple([i for i, tok in enumerate(doc) if tok._.wikt_mwe != "*"])
    #     lemmas = [doc[i].lemma_ for i in match_idx]
    #     result = ExampleType(
    #         example=doc,
    #         lemma=doc._.mwe_lemma,
    #         match_idx=match_idx,
    #         lemmas=lemmas,
    #         pos=doc._.mwe_pos,
    #     )

    #     return result

    def _sort_lemmas_by_rank(
        self,
        lemmas: list[str],
        rank_dict: Optional[dict[str, int]],
        ascending: bool = False,
    ):
        if rank_dict is None:
            return lemmas

        # Function to get the rank of a lemma
        def get_rank(lemma: str):
            if lemma in rank_dict:
                return rank_dict[lemma]
            elif lemma.isalpha():
                return 999999
            else:
                return 0  # Punctuation marks or numbers are rank 0

        # Sort the lemmas based on their ranks
        sorted_lemmas = sorted(lemmas, key=get_rank, reverse=not ascending)

        return sorted_lemmas

    def _doc_to_example_type(self, doc: Doc, mwe_label: str, rank_dict: Optional[dict[str, int]] = None):
        match_idx = tuple(
            [i for i, tok in enumerate(doc) if mwe_label in tok._.wikt_mwe]
        )
        lemmas = self._sort_lemmas_by_rank(
            [doc[i].lemma_ for i in match_idx], rank_dict
        )

        mwe_lemma = mwe_label.split(":")[1]
        mwe_pos = mwe_label.split(":")[2]
        result = ExampleType(
            example=doc,
            lemma=mwe_lemma,
            match_idx=match_idx,
            lemmas=lemmas,
            pos=mwe_pos,
        )

        return result

    def train_from_example(self, example: ExampleType):
        mwe_key = self._example_to_key(example)
        self.mwes[mwe_key]["lemmas"] = example["lemmas"]
        self.mwes[mwe_key]["pos"] = example["pos"]

        for filter_key in self._filters.keys():
            self._filters[filter_key].add_example(  # type: ignore
                self.mwes[mwe_key][filter_key], example
            )

    def train(self, examples: list[Doc], rank_dict: Optional[dict[str, int]] = None):
        for doc in examples:
            mwes_present: set[str] = {
                mwe
                for tok in doc
                if tok._.wikt_mwe != "*"
                for mwe in tok._.wikt_mwe.split("|")
            }
            for mwe in mwes_present:
                example = self._doc_to_example_type(doc, mwe, rank_dict)

                self.train_from_example(example)

    def apply_filters(
        self, doc: Doc, mwe: MWEType, match_idx: tuple[int, ...]
    ) -> tuple[bool, ...]:
        filter_results: tuple[bool, ...] = tuple(
            [
                self._filters[f_key].filter(mwe[f_key], doc, match_idx)  # type: ignore
                for f_key in self.active_filters[mwe["pos"]]
            ]
        )
        return filter_results

    def __call__(self, doc: Doc) -> Doc:
        predictions = ["*" for _ in doc]
        count: int = 0
        for mwe_key, mwe in self.mwes.items():
            lemmas = mwe["lemmas"]
            pos = mwe["pos"]
            token_lemmas = [tok.lemma_ for tok in doc]
            matches = (
                find_continuous_candidate_matches(lemmas, token_lemmas)
                if pos in self._data.continuous_POS
                else find_candidate_matches(lemmas, token_lemmas)
            )
            for match_idx in matches:
                if match_idx == ():
                    continue
                filter_results = self.apply_filters(doc, mwe, match_idx)
                if all(filter_results):
                    count += 1
                    for idx in match_idx:
                        label = str(count) + ":" + mwe_key

                        predictions[idx] = (
                            label
                            if predictions[idx] == "*"
                            else predictions[idx] + "|" + label
                        )
        for i, tok in enumerate(doc):
            tok._.wikt_mwe = predictions[i]

        return doc

    def to_disk(self, path: str, exclude: tuple[Any, ...] = tuple()):
        path_save: Path = ensure_path(path)
        if not path_save.exists():
            path_save.mkdir()

        srsly.write_json(  # type: ignore
            os.path.join(path, self._lang + "_data.json"), self._data.to_dict()
        )

    def from_disk(self, path: str, exclude: tuple[Any, ...] = tuple()):
        file_path = os.path.join(path, self._lang + "_data.json")
        path_save = ensure_path(file_path)
        data: MWEDetectorDataSerialized = srsly.read_json(path_save)  # type: ignore

        self._data.from_dict(data)
        return self
