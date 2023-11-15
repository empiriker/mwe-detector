import os

from spacy.lang.en import English
from spacy.lang.fr import French

from mwe_detector.model import MWEDetector

FN = os.path.join(os.path.dirname(__file__), "data")


assigns = ["token._.wikt_mwe"]
requires = ["token.lemma", "token.pos", "token.dep", "token.head", "token.morph"]


@French.factory(
    "mwe_detector",
    assigns=assigns,
    requires=requires,
)
def create_mwe_detector_fr(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk(FN)
    return mweDetector


@English.factory(
    "mwe_detector",
    assigns=assigns,
    requires=requires,
)
def create_mwe_detector_en(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk(FN)
    return mweDetector
