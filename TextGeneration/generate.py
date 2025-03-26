"""Code for calling the generating a text."""

from sys import argv

from TextGeneration.utils.files import json_to_schema, schema_to_json
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import InputSchema, OutputSchema
import pickle
import random
import sys


def choose_next_word(choices):
    """
    input: choices = all the possible choices based on the model
    This function choose the next word based on probability
    Note that it won't calculate the probabilities for n-grams occurring fewer than 6 times
    it returns the next most probable word
    """
    if not choices:
        return None
    total = sum(choices.values())
    probabilities = []
    for count in choices.values():
        if count < 6:
            probability = sys.float_info.epsilon
        else:
            probability = count / total
        probabilities.append(probability)

    if all(x == sys.float_info.epsilon for x in probabilities):
        return "</TEXT>"
    else:
        return random.choices(list(choices.keys()), probabilities)[0]


def generate_text(ngram_model, text, length=10, n=3):
    """
    inputs:
        ngram_model : the model that we have created
        text : input text
        length : the length of the generating text
        n : max size of ngram

        output : generated text
    """
    words = text.lower().split()

    for _ in range(length):
        # Get last (n-1) words for 3gram
        prefix = tuple(words[-(n - 1):])

        # Try 3-gram → 2-gram → 1-gram
        for i in range(n, 0, -1):
            choices = ngram_model[i].get(prefix[-(i - 1):], {})
            next_word = choose_next_word(choices)
            if next_word:
                words.append(next_word)
                # Stop if we find a valid word
                break
        else:
            # Stop if no valid ngram is found
            break

    return ' '.join(words)


def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """
    # Reading input data
    input_schema = json_to_schema(file_str_path=file_str_path, input_schema=InputSchema)
    for input_text in input_schema.texts:
        test_text = Preprocessor.clean(text=input_text)

    # load the model
    with open("3gram_model.pkl", "rb") as f:
        ngram_model = pickle.load(f)

    # generate text
    text = "only use"
    generated_text = generate_text(ngram_model, text, length=1)
    size_generated_words = 1

    # continue generating until the end condition is met
    while (size_generated_words < 50) and not (generated_text.endswith('</TEXT>')):
        generated_text = generate_text(ngram_model, generated_text, length=1)
        size_generated_words += 1

    # Printing generated texts
    output_schema = OutputSchema(generated_texts=generated_text)
    schema_to_json(file_path=input_schema.output_file, schema=output_schema)


if __name__ == "__main__":
    main_generate(file_str_path=argv[1])
