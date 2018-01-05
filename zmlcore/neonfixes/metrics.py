"""
Portions (C) Copyright 2017 Synacor, Inc.
Created: 9/22/2017, Michael Toutonghi

This and other files in this directory are fixes for the Neon library, typically in the form of a subclass, sometimes
a replacement

Some portions of this file are copyrighted by Intel and used without usage restriction in accordance
with the Apache 2.0 License, under which Intel Nervana Neon is licensed:
https://github.com/NervanaSystems/neon/blob/master/LICENSE
"""
import numpy as np
from neon.transforms.cost import Metric
from neon.transforms.activation import Softmax


# This fixes the bug where both the inputs and the target were passed to the underlying
# metrics
class MultiMetric(Metric):
    """
    A wrapper Metric which can be used with Tree models which have more than
    one output.  Tree models have tuples of tensors, one tensor per output.
    Wrapping a Metric with a MultiMetric ensures that the metric sees only one
    of those tensors in the output tuple instead of all of them.
    """

    def __init__(self, metric, index):
        """
        Args:
            metric (Metric): Metric to apply in this multi-output context
            index (integer): The index into the model's output tuple to apply
                             the metric to
        """
        self.metric = metric
        self.index = index
        super(MultiMetric, self).__init__()

    def __call__(self, y, t, *args, **kwargs):
        """
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy array : Returns the log loss  metric in numpy array,
                         [LogLoss]
        """
        return self.metric(y[self.index], t[self.index], *args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.metric, key)


class AverageLogLoss(Metric):

    """
    Average log loss across a number of non-exclusive classes with optional cost for false positive.
    """

    def __init__(self, false_positive=True):
        self.correctProbs = self.be.iobuf(1)
        self.metric_names = ['LogLoss']
        self.false_positive = false_positive
        super(AverageLogLoss, self).__init__()

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy array : Returns the log loss  metric in numpy array,
                         [LogLoss]
        """
        if self.false_positive:
            self.correctProbs[:] = -self.be.mean(self.be.safelog(y) * t + (self.be.safelog(1 - y) * (1 - t)), axis=0)
        else:
            self.correctProbs[:] = -self.be.mean(self.be.safelog(y) * t, axis=0)
        return np.array(self.correctProbs.get()[:, calcrange].mean())


