# A SpaCy MWE identification pipeline component

This component identifies multiword expressions (MWEs) in SpaCy documents and makes the output available at `token._.mwe_wikt`. The component, its underlying data and training are described in [Überrück-Fries et al. (2024)](https://aclanthology.org/2024.nlp4call-1.19/).

The component has been evaluated on the [Deep-Sequoia corpus](https://deep-sequoia.inria.fr/) and reached an F1-score of 0.776. Further details on the evaluation procedure and performance can also be found in the paper.

Currently MWE identification is supported only for French.

## Installation

1. Clone repository and install via

```bash
pip install .
```

2. Install directly from GitHub via

```bash
pip install git+https://github.com/empiriker/mwe-detector.git
```

## Usage

To identify MWEs, you'll need to import and add the pipeline to your SpaCy model:

```python
import mwe_detector.pipeline
import spacy

nlp = spacy.load("fr_core_news_sm")
nlp.add_pipe("mwe_detector")

doc = nlp("L'identification des expressions polylexicales va bon train.")

print([tok._.wikt_mwe for tok in doc])
# ['*', '*', '*', '*', '*', '1:aller bon train:VERB', '1:aller bon train:VERB', '1:aller bon train:VERB', '*']
```

The model will return a `wikt_mwe` label per token. If a token is not part of an MWE, the label is `*`. If a token is part of an MWE, it will receive a label in the format `[Number of MWE in doc, 1-indexed, integer]:[Lemma of MWE]:[POS of MWE]`. If a token is part of multiple MWEs, the different labels are separated by `|`.

## Development

To install the development dependencies, clone the repository and run

```bash
pip install .[dev]
```

### Train

Train the model on your own training data, specified in [config.py](./mwe_detector/config.py) with

```bash
python train.py --lang_code fr
```

## Data

This repository contains [data](./data/) that has been used in training and evaluation of the pipeline.

- [fr_train_wiktionary.cupt](./data/fr_train_wiktionary.cupt) contains example sentences extracted from the [French Wiktionary](https://fr.wiktionary.org). The sentences have been processed with SpaCy and converted to the cupt format. An additional column `WIKT:MWE` has been added. These labels are not exhaustive, i.e. not all MWEs that could be annotated are annotated.
- [fr_test_sequoia.cupt](./data/fr_test_sequoia.cupt) contains the original [Deep-Sequoia corpus](https://gitlab.inria.fr/sequoia/deep-sequoia/-/blob/master/tags/sequoia-9.2/sequoia-ud.parseme.frsemcor?ref_type=heads) but replaces the original annotations `PARSEME:MWE` and `FRSEMCOR:NOUN` with `WIKT:MWE` as described in Überrück-Fries et al. (2024).
- [fr_rank.json](./data/fr_rank.json) is a rank dictionary derived from the [Lexique383](http://www.lexique.org/databases/Lexique383/) word list. It serves to store an MWEs constituent lemmas by inverse order of frequency, optimizing the search for MWE candidates.

## Acknowledgements

This work was funded by an internship grant form the Graduate School in Computer Science of the Paris-Saclay University, as well as by the French Agence Nationale pour la Recherche, through the SELEXINI project (ANR-21-CE23-0033-01).

## License

The file [fr_test_sequoia.cupt](./data/fr_test_sequoia.cupt) is licensed with [LGPLLR](./LICENSE-LGPLLR). All other files in this repository are licensed with [CC BY-SA 4.0](./LICENSE).
