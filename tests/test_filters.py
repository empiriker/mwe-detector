import unittest

# from multiset import Multiset
from spacy import load

# from spacy.tokens import Doc
from mwe_detector.filters import F1, F2, F3, F4, F5, F6, F7, ExampleType

nlp = load("en_core_web_sm")


class TestF1(unittest.TestCase):
    def setUp(self):
        self.filter = F1()
        self.doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        self.doc2 = nlp("Time flies like an arrow; fruit flies like a banana.")

    def test_add_example(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example1",
            lemmas=["quick", "brown", "fox"],
            example=self.doc1,
            match_idx=(1, 2, 3),
            pos="VERB",
        )

        self.filter.add_example(data, example)
        self.assertEqual(len(data), 1)  # type: ignore
        self.assertEqual(data[0], ["ADJ", "ADJ", "NOUN"])  # type: ignore

    def test_filter(self):
        data = self.filter.default_data()
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


class TestF2(unittest.TestCase):
    def setUp(self):
        self.filter = F2()
        self.doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        self.doc2 = nlp("Time flies like an arrow; fruit flies like a banana.")

    def test_add_example(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example2",
            lemmas=["jumps", "over", "the"],
            example=self.doc1,
            match_idx=(4, 5, 6),
            pos="VERB",
        )

        self.filter.add_example(data, example)
        self.assertEqual(len(data), 1)  # type: ignore
        self.assertEqual(data[0], ["VERB", "ADP", "DET"])  # type: ignore

    def test_filter(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example2",
            lemmas=["jumps", "over", "the"],
            example=self.doc1,
            match_idx=(4, 5, 6),
            pos="VERB",
        )
        self.filter.add_example(data, example)

        match_idx = (4, 5, 6)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertTrue(result)

        match_idx = (4, 6, 5)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertFalse(result)

        match_idx = (2, 3, 5)
        result = self.filter.filter(data, self.doc2, match_idx)
        self.assertFalse(result)


class TestF3(unittest.TestCase):
    def setUp(self):
        self.filter = F3()
        self.doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        self.doc2 = nlp("Time flies like an arrow; fruit flies like a banana.")

    def test_add_example(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example3",
            lemmas=["The", "fox", "jumps"],
            example=self.doc1,
            match_idx=(0, 3, 4),
            pos="VERB",
        )

        self.filter.add_example(data, example)
        self.assertEqual(len(data), 1)  # type: ignore
        # Here we expect all the POS tags from "The" to "jumps", inclusive
        self.assertEqual(data[0], ["DET", "ADJ", "ADJ", "NOUN", "VERB"])  # type: ignore

    def test_filter(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example3",
            lemmas=["The", "fox", "jumps"],
            example=self.doc1,
            match_idx=(0, 3, 4),
            pos="VERB",
        )
        self.filter.add_example(data, example)

        match_idx = (0, 3, 4)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertTrue(result)

        match_idx = (0, 4, 3)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertTrue(result)

        match_idx = (0, 2, 4)
        result = self.filter.filter(data, self.doc2, match_idx)
        self.assertFalse(result)


class TestF4(unittest.TestCase):
    def setUp(self):
        self.filter = F4()
        self.doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        self.doc2 = nlp("Time flies like an arrow; fruit flies like a banana.")

    def test_add_example(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example4",
            lemmas=["The", "jumps", "dog"],
            example=self.doc1,
            match_idx=(0, 4, 8),
            pos="VERB",
        )

        self.filter.add_example(data, example)
        self.assertEqual(len(data), 1)  # type: ignore
        # The largest discontinuity here is from "jumps" to "dog" which is 4
        self.assertEqual(data[0], 4)  # type: ignore

    def test_filter(self):
        data = self.filter.default_data()
        example = ExampleType(
            lemma="example4",
            lemmas=["The", "jumps", "dog"],
            example=self.doc1,
            match_idx=(0, 4, 8),
            pos="VERB",
        )
        self.filter.add_example(data, example)

        # Test a positive match with same discontinuity
        match_idx = (0, 4, 8)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertTrue(result)

        # Test a negative match with a larger discontinuity
        match_idx = (0, 3, 8)
        result = self.filter.filter(data, self.doc1, match_idx)
        self.assertFalse(result)

        # Test a negative match with a different document and larger discontinuity
        match_idx = (0, 2, 8)
        result = self.filter.filter(data, self.doc2, match_idx)
        self.assertFalse(result)


class TestF5(unittest.TestCase):
    def setUp(self):
        self.filter = F5()
        self.doc = nlp(
            "Time flies like an arrow; fruit flies like a banana. People like these flies."
        )

    def test_filter(self):
        # Multiple matches for the lemmata "flies", "like", and "banana"
        match_idx1 = (1, 2, 10)  # "flies", "like", "banana" (Discontinuity of 7)
        match_idx2 = (7, 8, 10)  # "flies", "like", "banana" (Discontinuity of 1)

        # Only the second match should be approved by the filter since it has the least discontinuity
        self.assertFalse(self.filter.filter(None, self.doc, match_idx1))  # type: ignore
        self.assertTrue(self.filter.filter(None, self.doc, match_idx2))  # type: ignore


class TestF6(unittest.TestCase):
    def setUp(self):
        self.filter = F6()
        self.doc = nlp("John, who lives in New York, likes apples.")

    def test_two_token_connected(self):
        # "John" and "likes" are connected (subject-verb relation)
        match_idx = (0, 8)
        self.assertTrue(self.filter.filter(None, self.doc, match_idx))  # type: ignore

    def test_two_token_disconnected(self):
        # "John" and "apples" are not directly connected
        match_idx = (0, 9)
        self.assertFalse(self.filter.filter(None, self.doc, match_idx))  # type: ignore

    def test_two_token_grandparent(self):
        # "John" and "who" have a grandparent relation
        match_idx = (0, 2)
        self.assertTrue(self.filter.filter(None, self.doc, match_idx))  # type: ignore

    def test_two_token_greatgrandparent(self):
        # "John" and "New" have a grandparent relation
        match_idx = (0, 5)
        self.assertFalse(self.filter.filter(None, self.doc, match_idx))  # type: ignore

    def test_multi_token_connected(self):
        # "John", "lives", "in", "New", "York" form a connected subgraph
        match_idx = (0, 3, 4, 5, 6)
        self.assertTrue(self.filter.filter(None, self.doc, match_idx))  # type: ignore

        # "John", "likes", "apples" do not form a connected subgraph
        match_idx = (0, 8, 9)
        self.assertTrue(self.filter.filter(None, self.doc, match_idx))  # type: ignore

    def test_multi_token_disconnected(self):
        # "John", "lives", "in", "apples" do not form a connected subgraph
        match_idx = (0, 3, 4, 9)
        self.assertFalse(self.filter.filter(None, self.doc, match_idx))  # type: ignore


class TestF7(unittest.TestCase):
    def setUp(self):
        self.filter = F7()
        self.doc = nlp("The quick brown fox jumps over the lazy dogs.")

    def test_no_nouns(self):
        # "quick" and "brown" have no nouns
        match_idx = (1, 2)
        self.assertTrue(self.filter.filter([], self.doc, match_idx))  # type: ignore

    def test_multiple_nouns(self):
        # "fox" and "dogs" have multiple nouns
        match_idx = (3, 8)
        self.assertTrue(self.filter.filter([], self.doc, match_idx))  # type: ignore

    def test_one_noun_observed_inflection(self):
        # Let's train the filter with the inflection of "dogs"
        example = ExampleType(
            lemma="test", lemmas=["dogs"], example=self.doc, match_idx=(8,), pos="NOUN"
        )
        noun_morphs = self.filter.default_data()
        self.filter.add_example(noun_morphs, example)  # type: ignore

        # "dogs" has one noun with observed inflection
        match_idx = (8,)
        self.assertTrue(self.filter.filter(noun_morphs, self.doc, match_idx))

    def test_one_noun_unobserved_inflection(self):
        # Let's train the filter with the inflection of "dogs"
        example = ExampleType(
            lemma="test", lemmas=["dogs"], example=self.doc, match_idx=(8,), pos="NOUN"
        )
        noun_morphs = self.filter.default_data()
        self.filter.add_example(noun_morphs, example)  # type: ignore

        # "fox" has one noun with unobserved inflection
        match_idx = (3,)
        self.assertFalse(self.filter.filter(noun_morphs, self.doc, match_idx))

    def test_one_noun_no_observed_inflection(self):
        # We did not train the filter with the inflection of "fox"
        match_idx = (3,)
        self.assertFalse(self.filter.filter([], self.doc, match_idx))  # type: ignore


if __name__ == "__main__":
    unittest.main()
