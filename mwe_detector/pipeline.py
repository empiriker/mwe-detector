import os

# from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.language import Language

from mwe_detector.model import MWEDetector

FN = os.path.join(os.path.dirname(__file__), "data")


assigns = ["token._.wikt_mwe"]
requires = ["token.lemma", "token.pos", "token.dep", "token.head", "token.morph"]


@French.factory(  # type: ignore
    "mwe_detector",
    assigns=assigns,
    requires=requires,
)
def create_mwe_detector_fr(nlp: Language, name: str):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk(FN)
    return mweDetector


# @English.factory(  # type: ignore
#     "mwe_detector",
#     assigns=assigns,
#     requires=requires,
# )
# def create_mwe_detector_en(nlp: Language, name: str):
#     mweDetector = MWEDetector(nlp)
#     mweDetector.from_disk(FN)
#     return mweDetector
