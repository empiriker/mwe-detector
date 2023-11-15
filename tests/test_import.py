import pytest
import spacy

import mwe_detector.pipeline
from mwe_detector.model import MWEDetector


def test_custom_pipeline_fr():
    nlp = spacy.load("fr_core_news_sm")
    nlp.add_pipe("mwe_detector")

    # Check that the custom pipeline component is present
    assert "mwe_detector" in nlp.pipe_names

    # Check that the custom pipeline component is an instance of MWEDetector
    mwe_detector = nlp.get_pipe("mwe_detector")
    assert isinstance(mwe_detector, MWEDetector)

    # Check that the data is available in the pipeline
    assert mwe_detector.mwes

    # Test the functionality of the custom pipeline component
    doc = nlp("This is a sample sentence for testing the mwe_detector.")
    tok = doc[0]

    # Check if the document has the custom extension attributes
    assert hasattr(tok._, "wikt_mwe")


def test_custom_pipeline_en():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("mwe_detector")

    # Check that the custom pipeline component is present
    assert "mwe_detector" in nlp.pipe_names

    # Check that the custom pipeline component is an instance of MWEDetector
    mwe_detector = nlp.get_pipe("mwe_detector")
    assert isinstance(mwe_detector, MWEDetector)

    # Check that the data is available in the pipeline
    assert mwe_detector.mwes

    # Test the functionality of the custom pipeline component
    doc = nlp("This is a sample sentence for testing the mwe_detector.")
    tok = doc[0]

    # Check if the document has the custom extension attributes
    assert hasattr(tok._, "wikt_mwe")
