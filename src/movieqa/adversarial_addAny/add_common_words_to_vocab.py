import os
import sys

present_path = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.join(present_path, '../../'))

import core.util as util
import movieqa.data_conf as data_conf

glove = util.loadGloveModel(data_conf.PRETRAINED_EMBEDDINGS_PATH)

#vectors, vocab = util.load_embeddings(data_conf.EMBEDDING_DIR)

util.restore_vocab(data_conf.EMBEDDING_DIR)

print("Restored vocab")

#rev_vocab = dict(zip(vocab.values(), vocab.keys()))
#print("Current vocabulary %s with %d entries" % (str(rev_vocab), len(rev_vocab)))

filename = "adversarial_addAny/common_english.txt"
fin = open(filename, encoding="utf8")
for line in fin:
    word = line.replace('\n', '')
    print("get word vector for %s" % word)
    vec = util.get_word_vector(glove, word, data_conf.EMBEDDING_SIZE)

vsize = util.save_embeddings(data_conf.EMBEDDING_DIR, data_conf.EMBEDDING_SIZE)
print("New vocabulary size %d" % vsize)
