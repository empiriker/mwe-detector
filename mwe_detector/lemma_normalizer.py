from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.language import Language


class LemmaNormalizer:
    def __init__(self, lemma_table):
        self.lemma_table = lemma_table

    def __call__(self, doc):
        for token in doc:
            # Overwrite the token.lemma_ if there's an entry in the data
            token.lemma_ = self.lemma_table.get(token.lemma_, token.lemma_)
        return doc


@English.factory("lemma_normalizer", assigns=["token.lemma"], requires=["token.lemma"])
def create_en_normalizer(nlp, name):
    return LemmaNormalizer({"'ve": "have"})


@French.factory("lemma_normalizer")
def create_fr_normalizer(nlp, name):
    return LemmaNormalizer({"d'": "de"})


@Language.factory("lemma_normalizer", assigns=["token.lemma"], requires=["token.lemma"])
def token_lemma_normalizer(nlp, name):
    return LemmaNormalizer({})
