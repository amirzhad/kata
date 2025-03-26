import random
import re
from collections import defaultdict

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

def build_ngram_model(words, n=3):
    ngram_model = defaultdict(list)

    for i in range(len(words)-n+1):
        prefix = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        ngram_model[prefix].append(next_word)

    return ngram_model


def generate_text(ngram_model, seed_text, length=10):
    words = seed_text.lower().split()
    n = len(list(ngram_model.keys())[0]) + 1  # detect ngram size

    for _ in range(length):
        prefix = tuple(words[-(n - 1):])  # get the last n-1 words
        possible_next_words = ngram_model.get(prefix, [])

        if not possible_next_words:
            break

        next_word = random.choice(possible_next_words)
        words.append(next_word)

    return ' '.join(words)

# sample curpus
text = "I love programming in Python. Python is great for data science. Machine Learning is fun."

# train the model
words = preprocess_text(text)

ngram_model = build_ngram_model(words, n=2)

seed_text = "is"
generated_text = generate_text(ngram_model, seed_text, length=1)
print(generated_text)