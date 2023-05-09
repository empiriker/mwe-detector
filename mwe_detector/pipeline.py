from spacy.lang.fr import French
from spacy.lang.en import English
from mwe_detector.model import MWEDetector


@French.factory("mwe_detector")
def create_mwe_detector(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk("mwe_detector/data")
    return mweDetector


@English.factory("mwe_detector")
def create_mwe_detector(nlp, name):
    mweDetector = MWEDetector(nlp)
    mweDetector.from_disk("mwe_detector/data")
    return mweDetector
