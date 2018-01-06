"""
created: 12/14/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

This handles vocabularies for processing text.

"""
import os
import re
import json
import numpy as np
from collections import OrderedDict
from zmlcore.licensed.datautils import ArrayFields


def clean_text(text):
    """
    replaces the occurrence of 3 or more "word" characters with 3 of those same characters, for example:
    (soooooo | soooo | sooooooooooo) -> sooo
    yesssss! -> yesss!
    :param text:
    :return:
    """
    RE_DUPS = re.compile(r"([\w!?])\1{3,}")
    return RE_DUPS.sub(r'\1\1\1', text)


class Vocabularies(object):
    _vocabularies = {}

    @staticmethod
    def gen_vocabulary(vocab_path, documents, regex, preserve_case=False, n_first_words=60, size=int(1.0E7), save=True):
        """
        :param vocab_dict:
        :param documents: list of strings of text of each document's content
        :param n_first_words: the number of words to use from the beginning of each document
        :param size:
        :param save:
        """
        vocab_dict = OrderedDict() # an ordered dict allows us to map from value to key by position
        stat_fields = ['total_count', 'doc_count', 'last_doc', 'avg_tf_idf', 'idf']
        stat = ArrayFields(np.zeros(len(stat_fields)), stat_fields)

        for text, i in zip(documents, range(1, len(documents) + 1)):
            text = clean_text(text)
            for w, _ in zip((s.group(0) if preserve_case else s.group(0).lower() for s in re.finditer(regex, text)),
                            range(n_first_words)):
                stat.array = vocab_dict.setdefault(w, np.zeros(len(stat_fields), np.float32))
                if stat.last_doc != i:
                    stat.last_doc = i
                    stat.doc_count += 1

                stat.total_count += 1

        # now, every entry has a total count and the count of documents it is in, so, we can
        # calculate an average tf*idf for every word in the collection, then we sort and prune if necessary
        # for our final vocabulary
        dc = len(documents)
        sort_by = []
        for v in vocab_dict.values():
            stat.array = v
            stat.idf = np.log(dc / stat.doc_count)
            stat.avg_tf_idf = (stat.total_count / stat.doc_count) * stat.idf
            sort_by += [-stat.avg_tf_idf]

        # get indexes to our words by average tfidf in descending order, then prune
        idxs = np.argsort(sort_by)[:size]
        keys = np.take(list(vocab_dict.keys()), idxs)

        # reorder and remake our dict into {word: (stats, word_input)}, no persisted stats for now
        vocab_dict = dict([(k, np.array([i])) for k, i in zip(keys, range(len(keys)))])

        Vocabularies._vocabularies[vocab_path] = vocab_dict
        if save:
            Vocabularies.save_vocabulary(vocab_path, vocab_dict)

        return vocab_dict

    @staticmethod
    def save_vocabulary(vocab_path, dict):
        # dump out the raw vocabulary file
        with open(vocab_path, 'w') as f:
            f.writelines([' '.join([k, *[str(v) for v in dict[k]]]) + '\n' for k in dict])

    @staticmethod
    def load_vocabulary(vocab_path):
        """
        path for word2vec or glove text embedding vocabularies
        :param vocab_path:
        :return:
        """
        vocab_path = os.path.abspath(vocab_path)
        assert vocab_path and vocab_path != ''
        vocab = Vocabularies._vocabularies.get(vocab_path, None)
        if vocab is None:
            try:
                vocab = {}
                print('loading word vectors from {} ... '.format(vocab_path), end='', flush=True)
                failures = 0
                with open(vocab_path, 'r') as f:
                    tokens = []
                    for line in f:
                        if len(line) > 0:
                            tokens = line.split()
                            try:
                                vocab[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
                            except Exception as e:
                                failures += 1
                if failures > 0:
                    print('failed to load {} word vectors... '.format(failures), end='')
                print('loaded {} words successfully'.format(len(vocab)))
                Vocabularies._vocabularies[vocab_path] = vocab
            except FileNotFoundError:
                vocab = None

        return vocab

