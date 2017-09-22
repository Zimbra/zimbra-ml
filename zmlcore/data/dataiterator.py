"""
created: 9/21/2017
(c) copyright 2017 Synacor, Inc

This is a simple iterator for training data that iterates over both the inputs and targets for training
neural networks. It expects the inputs and targets to be independent iterables of the same number of elements
and simply provides the interface Intel's Nervana Neon Model likes to see
"""
import numpy as np
from neon.data import NervanaDataIterator

class TrainingIterator(NervanaDataIterator):
    def __init__(self, inputs, targets=None, name=None):
        """
        Just an easy way to feed data to a neon model
        :param name:
        """
        assert len(inputs) == len(targets) or targets is None
        super(TrainingIterator, self).__init__(name=name)
        self.inputs = inputs
        self.targets = targets

    @property
    def nbatches(self):
        return len(self.inputs)

    @property
    def ndata(self):
        return self.nbatches

    def reset(self):
        pass

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        for x, y in zip(self.inputs, self.targets if not self.targets is None else self.inputs):
            yield (x, y)

    def shuffle(self):
        a = np.arange(len(self.inputs))
        np.random.shuffle(a)
        self.inputs[:] = self.be.take(self.inputs, a, axis=0)
        if not self.targets is None:
            self.targets[:] = self.be.take(self.targets, a, axis=0)