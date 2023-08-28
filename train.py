import argparse
import spacy
from spacy.tokens import Doc, Token

import os
import glob
from datetime import datetime


from mwe_detector.model import MWEDetector
from config import SPACY_MODEL, TRAIN_DATA_DIR

from load_cupt_to_spacy import load_cupt_to_spacy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model for MWE detection")

    parser.add_argument(
        "--lang_code",
        type=str,
        help="Language code of the train file.",
    )

    args = parser.parse_args()

    nlp = spacy.load(SPACY_MODEL[args.lang_code])
    if not Doc.has_extension("mwe_lemma"):
        Doc.set_extension("mwe_lemma", default="")
    if not Doc.has_extension("mwe_pos"):
        Doc.set_extension("mwe_pos", default="")
    if not Token.has_extension("wikt_mwe"):
        Token.set_extension("wikt_mwe", default="*")

    matching_files = glob.glob(
        os.path.join(TRAIN_DATA_DIR, f"{args.lang_code}_training_*.cupt")
    )

    # Function to extract timestamp from filename
    def extract_timestamp(filename):
        # Extract the date and time from the filename
        timestamp_str = (
            filename.split("_")[3] + "_" + filename.split("_")[4].split(".")[0]
        )

        # Convert string to datetime object
        return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

    # Sort files based on the timestamp
    latest_file = sorted(matching_files, key=extract_timestamp)[-1]

    train_file = latest_file

    train_data = load_cupt_to_spacy(train_file, nlp)

    mweDetector = MWEDetector(nlp)
    mweDetector.train(train_data)
    mweDetector.to_disk("mwe_detector/data")
