from spacy.lang.fr import French
from spacy.lang.en import English
from mwe_detector.model import MWEDetector

import os

FN = os.path.join(os.path.dirname(__file__), "data")


@French.factory("mwe_detector")
def create_mwe_detector(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk(FN)
    return mweDetector


@English.factory("mwe_detector")
def create_mwe_detector(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk(FN)
    return mweDetector
