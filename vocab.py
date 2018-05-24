# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


class CharEmbed():
    def __init__(self, vector, count=1):
        assert type(vector) == np.ndarray
        self.vector = vector
        self.count = float(count)

    def avg(self):
        ret = self.vector/float(self.count)
        assert type(ret) == np.ndarray
        return ret

    def get_vec(self): return self.vector
    def get_count(self): return self.count

    def __str__(self):
        return 'Count: {}, Vector: {}'.format(self.get_count(), self.get_vec())


def get_glove(glove_dim, glove_path='glove.6B.100d.txt'):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.
    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path
    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded
    char_vocab_size = 26 # letters in alphabet + _PAD, _UNK tokens
    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    #char_emb_matrix = np.zeros((char_vocab_size+len(_START_VOCAB), glove_dim)) # is this the right dimension for char_emb_matrix
    word2id = {}
    id2word = {}
    char2id = {}
    id2char = {}

    char_emb_matrix_temp = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)
        char_emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    #cdx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
    #    char2id[word] = cdx
    #    id2char[cdx] = word
        #char_emb_matrix_temp[cdx] = CharEmbed(char_emb_matrix[cdx], 1)
        idx += 1
    #    cdx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            chars = list(word)

            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))

            # how to build the char embedding matrix...?
        #    for c in chars:
        #        if c not in char2id and c.isalpha():
        #            char2id[c] = cdx
        #            id2char[cdx] = c
        #            char_emb_matrix_temp[cdx] = CharEmbed(np.asarray(vector))
        #            cdx += 1
                ## THESE TWO CASES CAN BE COMBINED BUT LEAVING IT TO EXPLICITLY DEAL
                ## WITH UNK TOKENS
        #        elif c.isalpha():
        #            c_id = char2id[c]
        #            prev_c_emb_vec, prev_c_emb_count = char_emb_matrix_temp[c_id].get_vec(), char_emb_matrix_temp[c_id].get_count()
        #            char_emb_matrix_temp[c_id] = CharEmbed(prev_c_emb_vec+vector, prev_c_emb_count+1)
        #        else: # char is UNK
        #            prev_unk_vec, prev_unk_count = char_emb_matrix_temp[1].get_vec(), char_emb_matrix_temp[1].get_count()
        #            char_emb_matrix_temp[1] = CharEmbed(prev_unk_vec+vector, prev_unk_count+1)


            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    # This generates the average vectors for the chars
    #    key is index, value is CharEmbed object
    for key, value in char_emb_matrix_temp.iteritems(): char_emb_matrix[key, :] = value.avg()

    final_vocab_size = vocab_size + len(_START_VOCAB)
    final_char_size = char_vocab_size + len(_START_VOCAB)

    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    #assert len(char2id) == final_char_size
    #assert len(id2char) == final_char_size
    assert idx == final_vocab_size
    #assert cdx == final_char_size

    return emb_matrix, word2id, id2word, char_emb_matrix, char2id, id2char
