# Functional libraries
from spacy.tokens import Doc, Token
from spacy.language import Language
import srsly
from spacy.util import ensure_path
from .utils import find_candidate_matches, find_continuous_candidate_matches

# Utilities
from collections import defaultdict
import os

# Type hints
from typing import List, TypedDict, Tuple, DefaultDict
from pathlib import Path


from mwe_detector.filters import (
    F1Data,
    F1,
    F2Data,
    F2,
    F3Data,
    F3,
    F4Data,
    F4,
    F5Data,
    F5,
    F6Data,
    F6,
    F7Data,
    F7,
    F8Data,
    F8,
    ExampleType,
)


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
    lemmas: List[str]
    f1: F1Data
    f2: F2Data
    f3: F3Data
    f4: F4Data
    f5: F5Data
    f6: F6Data
    f7: F7Data
    f8: F8Data


class MWEDetectorData:
    def __init__(self):
        self.mwes: DefaultDict[str, MWEType] = defaultdict(
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
        self.active_filters = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        self.continuous_POS = ["ADJ", "ADV", "ADP", "CONJ", "INTJ", "NOUN", "PROPN"]

    def to_dict(self):
        return {"mwes": self.mwes, "active_filters": self.active_filters}

    def from_dict(self, data):
        self.mwes = data["mwes"]
        self.active_filters = data["active_filters"]

    def __getitem__(self, key):
        return self.mwes[key]

    def __setitem__(self, key, value):
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
    def active_filters(self, new_active_filters):
        self._data.active_filters = new_active_filters

    def _example_to_key(self, example: ExampleType):
        # lemmas = example["lemmas"]
        # lemmas.sort()
        return example["lemma"] + ":" + example["pos"]

    def _doc_to_example_type(self, doc: Doc):
        match_idx = tuple([i for i, tok in enumerate(doc) if tok._.wikt_mwe != "*"])
        lemmas = [doc[i].lemma_ for i in match_idx]
        result = ExampleType(
            example=doc,
            lemma=doc._.mwe_lemma,
            match_idx=match_idx,
            lemmas=lemmas,
            pos=doc._.mwe_pos,
        )

        return result

    def train_from_example(self, example: ExampleType):
        mwe_key = self._example_to_key(example)
        self.mwes[mwe_key]["lemmas"] = example["lemmas"]
        self.mwes[mwe_key]["pos"] = example["pos"]

        for filter_key in self._filters.keys():
            self._filters[filter_key].add_example(
                self.mwes[mwe_key][filter_key], example
            )

    def train(self, examples: List[Doc]):
        if not Doc.has_extension("mwe_lemma"):
            Doc.set_extension("mwe_lemma", default="")
        if not Doc.has_extension("mwe_pos"):
            Doc.set_extension("mwe_pos", default="")

        for example in examples:
            example_as_example_type = self._doc_to_example_type(example)
            self.train_from_example(example_as_example_type)

    # def _find_candidate_matches(self, lemmas: List[str], sent_doc: Doc):
    #     single_matches : List[List[int,]] = []
    #     for lemma in lemmas:
    #         matched_tokens = list(
    #             filter(lambda x: x.lemma_.lower() == lemma.lower(), sent_doc)
    #         )
    #         if len(matched_tokens) == 0:
    #             return []
    #         single_matches.append([tok.i for tok in matched_tokens])

    #     return list(product(*single_matches))

    def apply_filters(
        self, doc: Doc, mwe, match_idx
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        filter_results = tuple(
            [
                self._filters[f_key].filter(mwe[f_key], doc, match_idx)
                for f_key in self.active_filters
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
                            else predictions[idx] + ", " + label
                        )
        for i, tok in enumerate(doc):
            tok._.wikt_mwe = predictions[i]

        return doc

    def to_disk(self, path: str, exclude=tuple()):
        path_save: Path = ensure_path(path)
        if not path_save.exists():
            path_save.mkdir()

        srsly.write_json(
            os.path.join(path, self._lang + "_data.json"), self._data.to_dict()
        )

    def from_disk(self, path: str, exclude=tuple()):
        file_path = os.path.join(path, self._lang + "_data.json")
        path_save = ensure_path(file_path)
        data = srsly.read_json(path_save)

        self._data.from_dict(data)
        return self
