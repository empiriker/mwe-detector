import pytest
import spacy

import mwe_detector.lemma_normalizer


def test_en_base_lemmatizer():
    nlp = spacy.load("en_core_web_sm")
    test_sentence = "I should've done this earlier."
    expected_lemmas = ["I", "should", "'ve", "do", "this", "early", "."]
    doc = nlp(test_sentence)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == expected_lemmas


def test_en_normalizer():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("lemma_normalizer")
    test_sentence = "I should've done this earlier."
    expected_lemmas = ["I", "should", "have", "do", "this", "early", "."]
    doc = nlp(test_sentence)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == expected_lemmas


def test_fr_base_lemmatizer():
    nlp = spacy.load("fr_core_news_sm")
    test_sentence = "D'aller au Ciel et d'être riche?"
    expected_lemmas = [
        "de",
        "aller",
        "au",
        "ciel",
        "et",
        "de",
        "être",
        "riche",
        "?",
    ]
    doc = nlp(test_sentence)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == expected_lemmas


def test_fr_normalizer():
    nlp = spacy.load("fr_core_news_sm")
    nlp.add_pipe("lemma_normalizer")
    test_sentence = "C'est l'histoire d'une ville."
    expected_lemmas = ["ce", "être", "le", "histoire", "de", "un", "ville", "."]
    doc = nlp(test_sentence)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == expected_lemmas


def test_no_normalization():
    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("lemma_normalizer")
    test_sentence = "Dieser Satz hat keine besonderen Lemmas."
    expected_lemmas = [
        "dieser",
        "Satz",
        "haben",
        "kein",
        "besonderer",
        "Lemmas",
        "--",
    ]
    doc = nlp(test_sentence)
    lemmas = [token.lemma_ for token in doc]
    assert lemmas == expected_lemmas
