import random
from collections import defaultdict, Counter
from pathlib import Path
from TextGeneration.utils.preprocessor import Preprocessor
import sys

# Global variables
max_generated_text = 50
probability_calculate_limit = 6
ngram_size = 3

def read_all_files_as_text(directory, encoding="utf-8"):
    """
    a function that reads all the files and in a given folder
    """
    combined_text = ""

    for file_path in Path(directory).iterdir():
        # Ensure it's a file, not a subdirectory
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    # Separate file contents with newlines
                    combined_text += file.read() + "\n\n"
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

    return combined_text


def preprocess_text(document):
    """
    The preprocessing fucntion

    """
    preprocess = Preprocessor()
    document = preprocess.clean(document)
    document = document.replace("\n", " ")

    document = document.strip()
    document = "<TEXT> " + document
    document = document + " </TEXT>"

    document_words = document.split()
    return document_words


def build_ngram(words, n=ngram_size):
    """
    This function takes a tokenized text (with space) and build all the possible ngrams based on given n
    default value for n=3
    example: if n = 3 then this function will return 3-gram, 2-gram and 1-gram
    """

    # Store 1-gram, 2-gram, 3-gram
    ngram_model = {i: defaultdict(Counter) for i in range(1, n + 1)}

    for i in range(len(words)):
        for j in range(1, n + 1):  # Build all n-grams up to n
            if i + j <= len(words):
                prefix = tuple(words[i:i + j - 1])  # (n-1)-gram as key
                next_word = words[i + j - 1]  # Next word as value
                ngram_model[j][prefix][next_word] += 1  # Store count

    return ngram_model


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


def generate_text(ngram_model, text, length=10, n=ngram_size):
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


# read all files in the given directory
directory_path = "input_data"
documents = read_all_files_as_text(directory_path)

# pre-processing the data
words = preprocess_text(documents)

# creating the ngram models
ngram_model = build_ngram(words, n=ngram_size)

# create a sample text
sample_text = "use only"

# generate text
size_generated_words = 0
generated_text = generate_text(ngram_model, sample_text, length=1)
while (size_generated_words < max_generated_text) and not (generated_text.endswith('</TEXT>')):
    generated_text = generate_text(ngram_model, generated_text, length=1)
    size_generated_words += 1

# print the generated text
print(generated_text)

