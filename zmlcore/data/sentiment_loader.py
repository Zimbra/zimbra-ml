"""
created: 9/25/2017
(c) copyright 2017 Synacor, Inc

This is a loader for textual data to classify. It's initially intended to be used to sanity
check any classifying network used in the email classifier on the IMDB sentiment analysis
benchmark.
"""
import os
from neon import NervanaObject
from zmlcore.data.dataiterator import TrainingIterator
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
            train_neg_path = os.path.join(data_path, ['train', 'neg'])
            train_pos_path = os.path.join(data_path, ['train', 'pos'])
            test_neg_path = os.path.join(data_path, ['test', 'neg'])
            test_pos_path = os.path.join(data_path, ['test', 'pos'])
            train_files_neg = [f.strip() for f in next(os.walk(train_neg_path))[2]]
            train_files_pos = [f.strip() for f in next(os.walk(train_pos_path))[2]]
            test_files_neg = [f.strip() for f in next(os.walk(test_neg_path))[2]]
            test_files_pos = [f.strip() for f in next(os.walk(test_pos_path))[2]]

            train_x, train_t = self.load_classification(classifier,
                                                        train_files_neg,
                                                        self.be.array([0.0, 1.0]))
            x, t = self.load_classification(classifier,
                                            train_files_pos,
                                            self.be.array([1.0, 0.0]))
            train_x += x
            train_t += t
            self.train = TrainingIterator(train_x, train_t)

            test_x, test_t = self.load_classification(classifier,
                                                      test_files_neg,
                                                      self.be.array([0.0, 1.0]))
            x, t = self.load_classification(classifier,
                                            test_files_pos,
                                            self.be.array([1.0, 0.0]))
            test_x += x
            test_t += t
            self.test = TrainingIterator(test_x, test_t)

        else:
            print('Invalid IMDB data directory. specified directory should be',
                  'the root level of the IMDB dataset as published by Stanford University.')
            raise NotADirectoryError()

        super(SentimentLoader, self).__init__()

    def load_classification(self, classifier, path_list, target_tensor):
        x = []
        for fn in path_list:
            with open(fn, 'r') as t:
                x += classifier.text_to_nn_representation(str(t))
        t = [target_tensor for _ in range(len(x))]
        return x, t
