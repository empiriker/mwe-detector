import unittest

# from multiset import Multiset
from spacy import load

# from spacy.tokens import Doc
from mwe_detector.filters import F1, ExampleType

nlp = load("en_core_web_sm")


class TestF1(unittest.TestCase):
    def setUp(self):
        self.filter = F1()
        self.doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        self.doc2 = nlp("Time flies like an arrow; fruit flies like a banana.")

    def test_add_example(self):
        data = F1.default_data()
        example = ExampleType(
            lemma="example1",
            lemmas=["quick", "brown", "fox"],
            example=self.doc1,
            match_idx=(1, 2, 3),
            pos="VERB",
        )

        self.filter.add_example(data, example)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], ["ADJ", "ADJ", "NOUN"])

    def test_filter(self):
        data = F1.default_data()
        example = ExampleType(
            lemma="example1",
            lemmas=["quick", "brown", "fox"],
            example=self.doc1,
            match_idx=(1, 2, 3),
            pos="VERB",
        )
        self.filter.add_example(data, example)

        match_idx = (1, 2, 3)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertTrue(result)

        match_idx = (0, 1, 2)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertFalse(result)

        match_idx = (2, 3, 5)
        result = self.filter.filter(data, self.doc2, match_idx)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
