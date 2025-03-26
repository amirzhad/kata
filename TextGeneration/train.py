"""Code for calling the training of the model."""

from sys import argv

from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema
from collections import defaultdict, Counter
import pickle

def build_ngram(words, n=3):
    """
    This function takes a tokenized text (with space) and build all the possible ngrams based on given n
    default value for n=3
    example: if n = 3 then this function will return 3-gram, 2-gram and 1-gram
    """
    ngram_model = {i: defaultdict(Counter) for i in range(1, n + 1)}  # Store 1-gram, 2-gram, 3-gram

    for i in range(len(words)):
        for j in range(1, n + 1):  # Build all n-grams up to n
            if i + j <= len(words):
                prefix = tuple(words[i:i + j - 1])  # (n-1)-gram as key
                next_word = words[i + j - 1]  # Next word as value
                ngram_model[j][prefix][next_word] += 1  # Store count

    return ngram_model


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_schema = json_to_schema(
        file_str_path=file_str_path, input_schema=TrainingInputSchema
    )

    # pre-processing
    train_text = ""
    for training_line in read_dir(dir_path=training_schema.input_folder):
        # built-in preprocessing
        train_text += Preprocessor.clean(text=training_line)

    """
    This function do some preprocessing as demanded:

    - Never change the case of the input/output.
    - The new line character (\n) will mark the end of a string.
    - The new line character cannot be part of an n-gram so we replace them with space
    - Tokenize them based on white-spaces
    """

    # replacing end of line with space
    train_text = train_text.replace("\n", " ")

    # cleaning white space from start and end of the text
    train_text = train_text.strip()

    # adding <TEXT> to the beginning and end of the text
    train_text = "<TEXT> " + train_text
    train_text = train_text + " </TEXT>"

    # finally we tokenize the text based on white spaces
    words = train_text.split()

    # Train the model
    ngram_model = build_ngram(words, n=3)

    # save the model
    with open("3gram_model.pkl", "wb") as f:
        pickle.dump(ngram_model, f)


if __name__ == "__main__":
    main_train(file_str_path=argv[1])
