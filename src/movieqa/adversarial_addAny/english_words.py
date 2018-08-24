"""Get common English words from Brown Corpus."""
from nltk import FreqDist
from nltk.corpus import brown
import string

punctuation = set(string.punctuation) | set(['``', "''", "--"])


def get_common_words(outfile, num):
    freq_dist = FreqDist(w.lower() for w in brown.words() if w not in punctuation)
    vocab = [x[0] for x in freq_dist.most_common()[:num]]

    with open(outfile, "w") as of:
        for w in vocab:
            of.write(w + "\n")


if __name__ == '__main__':
    get_common_words('common_english.txt', 1000)
