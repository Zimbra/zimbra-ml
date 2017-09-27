"""
created: 9/25/2017
(c) copyright 2017 Synacor, Inc

This is a loader for textual data to classify. It's initially intended to be used to sanity
check any classifying network used in the email classifier on the IMDB sentiment analysis
benchmark.
"""
import os
import numpy as np
from neon import NervanaObject
from zmlcore.data.dataiterator import BatchIterator
from zmlcore.smartfolders.classifier import EmailClassifier


class SentimentLoader(NervanaObject):
    def __init__(self, classifier, data_path):
        """
        loads the IMDB dataset as published by Stanford for the following paper:
        http://www.aclweb.org/anthology/P11-1015
        """
        assert isinstance(classifier, EmailClassifier)
        if os.path.isdir(data_path):
            # get the folders in the directory
            # we care about train and test
            train_neg_path = os.path.join(os.path.join(data_path, 'train'), 'neg')
            train_pos_path = os.path.join(os.path.join(data_path, 'train'), 'pos')
            test_neg_path = os.path.join(os.path.join(data_path, 'test'), 'neg')
            test_pos_path = os.path.join(os.path.join(data_path, 'test'), 'pos')
            train_files_neg = [f.strip() for f in next(os.walk(train_neg_path))[2]]
            train_files_pos = [f.strip() for f in next(os.walk(train_pos_path))[2]]
            test_files_neg = [f.strip() for f in next(os.walk(test_neg_path))[2]]
            test_files_pos = [f.strip() for f in next(os.walk(test_pos_path))[2]]

            train_x, train_t = self.load_classification(classifier, train_neg_path,
                                                        train_files_neg,
                                                        np.array([0.0, 1.0]))

            x, t = self.load_classification(classifier, train_pos_path,
                                            train_files_pos,
                                            np.array([1.0, 0.0]))

            num_samples = len(train_files_neg) + len(train_files_pos)
            num_steps = int((len(train_x) + len(x)) / num_samples)
            train_x = np.array(train_x + x).reshape((num_samples, 1, num_steps, len(x[0])))
            train_t = np.array(train_t + t)
            # self.train = BatchIterator(train_x, train_t, steps=classifier.num_subject_words + classifier.num_body_words)
            self.train = BatchIterator(train_x, train_t)

            test_x, test_t = self.load_classification(classifier, test_neg_path,
                                                      test_files_neg,
                                                      np.array([0.0, 1.0]))
            x, t = self.load_classification(classifier, test_pos_path,
                                            test_files_pos,
                                            np.array([1.0, 0.0]))
            num_samples = len(test_files_neg) + len(test_files_pos)
            num_steps = int((len(test_x) + len(x)) / num_samples)
            test_x = np.array(test_x + x).reshape((num_samples, 1, num_steps, len(x[0])))
            test_t = np.array(test_t + t)
            # self.test = BatchIterator(test_x, test_t, steps=classifier.num_subject_words + classifier.num_body_words)
            self.test = BatchIterator(test_x, test_t)
        else:
            print('Invalid IMDB data directory {}. specified directory should be',
                  'the root level of the IMDB dataset as published by Stanford University.'.format(data_path))
            raise NotADirectoryError()

        super(SentimentLoader, self).__init__()

    def load_classification(self, classifier, base_path, file_list, targets):
        x = []
        for fn in file_list:
            with open(os.path.join(base_path, fn), 'r') as f:
                x += classifier.text_to_nn_representation(f.read())
        t = [targets for _ in range(len(file_list))]
        return x, t
