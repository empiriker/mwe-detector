from mwe_detector.model import MWEDetector
import mwe_detector.pipeline

import unittest
import spacy


class TestMWEDetector(unittest.TestCase):
    def test_custom_pipeline_fr(self):
        nlp = spacy.load("fr_core_news_sm")
        nlp.add_pipe("mwe_detector")

        # Check that the custom pipeline component is present
        self.assertIn("mwe_detector", nlp.pipe_names)

        # Check that the custom pipeline component is an instance of MWEDetector
        mwe_detector = nlp.get_pipe("mwe_detector")
        self.assertIsInstance(mwe_detector, MWEDetector)

        # Check that the data is available in the pipeline
        self.assertTrue(mwe_detector.mwes)

        # Test the functionality of the custom pipeline component
        # Add any necessary test data and expected results
        doc = nlp("This is a sample sentence for testing the mwe_detector.")
        tok = doc[0]

        # Check if the document has the custom extension attributes
        self.assertTrue(hasattr(tok._, "wikt_mwe"))

    def test_custom_pipeline_en(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("mwe_detector")

        # Check that the custom pipeline component is present
        self.assertIn("mwe_detector", nlp.pipe_names)

        # Check that the custom pipeline component is an instance of MWEDetector
        mwe_detector = nlp.get_pipe("mwe_detector")
        self.assertIsInstance(mwe_detector, MWEDetector)

        # Check that the data is available in the pipeline
        self.assertTrue(mwe_detector.mwes)

        # Test the functionality of the custom pipeline component
        # Add any necessary test data and expected results
        doc = nlp("This is a sample sentence for testing the mwe_detector.")
        tok = doc[0]

        # Check if the document has the custom extension attributes
        self.assertTrue(hasattr(tok._, "wikt_mwe"))


if __name__ == "__main__":
    unittest.main()
