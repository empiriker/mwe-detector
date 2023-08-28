import conllu
from spacy.tokens import Doc
from spacy.language import Language

from typing import List


def load_cupt_to_spacy(path: str, nlp: Language, mwe_column_name: str = "wikt:mwe"):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    # Parse the CoNLL-U data using the conllu package
    parsed_data = conllu.parse(data)

    docs: List[Doc] = []
    for cupt_sent in parsed_data:
        cupt_sent = [tok for tok in cupt_sent if "-" not in str(tok["id"])]
        words = [token["form"] for token in cupt_sent]
        spaces = [True if token["misc"] == None else False for token in cupt_sent]
        lemmas = [token["lemma"] for token in cupt_sent]
        pos = [token["upostag"] for token in cupt_sent]
        head = [
            token["head"] - 1 if token["head"] != 0 else i
            for i, token in enumerate(cupt_sent)
        ]
        dep = [token["deprel"] for token in cupt_sent]
        morph = [token["feats"] for token in cupt_sent]
        wikt_mwe = [token[mwe_column_name] for token in cupt_sent]
        doc = Doc(
            nlp.vocab,
            words=words,
            spaces=spaces,
            lemmas=lemmas,
            pos=pos,
            heads=head,
            deps=dep,
        )

        for i in range(len(doc)):
            doc[i]._.__setattr__(mwe_column_name.replace(":", "_"), wikt_mwe[i])
            doc[i].set_morph(morph[i])  # type: ignore
        docs.append(doc)

    return docs
