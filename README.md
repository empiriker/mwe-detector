# A SpaCy MWE identification pipeline component

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

## Acknowledgements

This work was funded by an internship grant form the Graduate School in Computer Science of the Paris-Saclay University, as well as by the French Agence Nationale pour la Recherche, through the SELEXINI project (ANR-21-CE23-0033-01).

## License

The file [fr_test_sequoia.cupt](./data/fr_test_sequoia.cupt) is licensed with [LGPLLR](./LICENSE-LGPLLR). All other files in this repository are licensed with [CC BY-SA 4.0](./LICENSE).
