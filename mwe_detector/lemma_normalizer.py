from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.language import Language
from spacy.tokens import Doc


class LemmaNormalizer:
    def __init__(self, lemma_table: dict[str, str]):
        self.lemma_table = lemma_table

    def __call__(self, doc: Doc):
        for token in doc:
            # Overwrite the token.lemma_ if there's an entry in the data
            token.lemma_ = self.lemma_table.get(token.lemma_, token.lemma_)
        return doc


@English.factory("lemma_normalizer", assigns=["token.lemma"], requires=["token.lemma"])  # type: ignore
def create_en_normalizer(nlp: Language, name: str):
    return LemmaNormalizer({"'ve": "have"})


@French.factory("lemma_normalizer")  # type: ignore
def create_fr_normalizer(nlp: Language, name: str):
    return LemmaNormalizer({"d'": "de"})


@Language.factory("lemma_normalizer", assigns=["token.lemma"], requires=["token.lemma"])  # type: ignore
def token_lemma_normalizer(nlp: Language, name: str):
    return LemmaNormalizer({})
