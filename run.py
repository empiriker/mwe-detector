train_file = (
    "/home/till/VSCode/mwe-detection/EntryDict2MWETrain/outputs/fr_dict_examples.cupt"
)

train_file = (
    "/home/till/VSCode/mwe-detection/EntryDict2MWETrain/outputs/fr_dict_examples.spacy"
)

test_file = "/home/till/VSCode/mwe-detection/data/parseme_corpus_fr/train.spacy"

import spacy
from spacy.tokens import Doc, DocBin, Token
from spacy.lang.fr import French

from mwe_detector.model import MWEDetector

nlp = spacy.load("fr_core_news_sm")
if not Doc.has_extension("mwe_lemma"):
    Doc.set_extension("mwe_lemma", default="")
if not Doc.has_extension("mwe_pos"):
    Doc.set_extension("mwe_pos", default="")
if not Token.has_extension("wikt_mwe"):
    Token.set_extension("wikt_mwe", default="*")

doc_bin = DocBin().from_disk(train_file)


examples = list(doc_bin.get_docs(nlp.vocab))


mweDetector = MWEDetector(nlp)
mweDetector.train(examples)
mweDetector.to_disk("mwe_detector")

mweDetector = MWEDetector(nlp)
mweDetector.from_disk("mwe_detector")

Token.set_extension("parseme_mwe", default="*")

doc_bin = DocBin().from_disk(test_file)
test_sents = list(doc_bin.get_docs(nlp.vocab))

# print(test_sentences[0].text)
# print(test_sentences[0][0].lemma_)
for test in test_sents:
    doc = mweDetector(test)
    print([tok._.wikt_mwe for tok in doc])

test_sents_plain = [sent.text for sent in test_sents]

# from spacy.language import Language


# # @Language.component("mwe_detector")
# # def mwe_detector(doc):
# #     return MWEDetector(doc)


# @French.factory("mwe_detector")
# def create_mwe_detector(nlp, name):
#     mweDetector = MWEDetector(nlp)
#     mweDetector.train(examples)
#     return mweDetector


# nlp = French()
# nlp.add_pipe("mwe_detector", last=True)

# doc = nlp(test_sents_plain[1])

# for sent in test_sents:
#     doc = nlp(sent.text)
#     for tok in doc:
#         if tok._.wikt_mwe:
#             print(tok.text, tok._.wikt_mwe)
