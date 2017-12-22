#
# (C) 2017, Michael Toutonghi - Licensed to Synacor for non-exclusive, unlimited use, reproduction, and derivation.
#
# Author: Michael Toutonghi
# Creation date: 1/12/2017
#


import numpy as np
from numbers import Number
from neon.layers.layer import Layer


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers.
    """
    if isinstance(xshape, Number):
        return (xshape, 1)
    else:
        if len(xshape) == 2:
            return xshape
        else:
            return (int(np.prod(xshape)), 1)


class NoisyDropout(Layer):

    """
        A layer to introduce dropout with noise between layers
    """

    def __init__(self, keep=0.6, noise_pct=0.01, noise_std=0.5, name=None, epsilon=1E-12):
        """
        :param keep: percentage to remain undisturbed
        :param noise_pct: percentage OF THOSE NOT KEPT to disturb by adding noise
        :param noise_std: percentage of each individual input that will be used as the std dev for random gaussian noise
        :param name:
        :param epsilon:
        """
        super(NoisyDropout, self).__init__(name=name)
        self.noise_std = noise_std
        self.keep = keep
        self.mask = None
        self.epsilon = epsilon
        self.owns_output = True
        self.owns_delta = True
        self.noise_pct = noise_pct
        self.caffe_mode = self.be.check_caffe_compat()
        if self.caffe_mode:
            self._train_scaling = 1.0 / keep
        else:
            self._train_scaling = 1.0

    def __str__(self):
        return "Noisy Dropout Layer : '%s' : %.3f%%" % self.name % self.noise_std

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(NoisyDropout, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            in_obj.next_layer = self
        return self

    def allocate(self, shared_outputs=None):
        super(NoisyDropout, self).allocate(shared_outputs)
        self.mask = self.be.iobuf(self.out_shape, parallelism=self.parallelism)

    def fprop(self, inputs, inference=False, alpha=None, beta=None):
        if self.outputs.shape != inputs.shape:
            inputs = inputs.reshape(self.outputs.shape)

        if not inference:
            self.outputs[:] = self.be.absolute(inputs * self.noise_std) + self.epsilon

            a = self.outputs.get()
            ash = a.shape
            a = np.reshape(a, tuple((a.size, 1)))
            a = np.random.normal(0, a)
            self.outputs[:] = np.reshape(a, ash)

            self.be.make_binary_mask(self.mask, 1 - self.noise_pct)
            self.outputs[:] = self.outputs * self.mask

            self.be.make_binary_mask(self.mask, self.keep)
            self.outputs[:] = self.be.equal(self.mask, 0) * self.outputs + (self.mask * inputs * self._train_scaling)
        else:
            if not self.caffe_mode:
                self.outputs[:] = inputs * self.keep
            else:
                self.outputs[:] = inputs

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if not self.deltas:
            self.deltas = error
        self.deltas[:] = self.mask * error * alpha * self._train_scaling + beta * error
        return self.deltas


class OutputDeltaBuffer(Layer):
    """
    A buffer layer to filter both fprop and bprop of inputs and outputs through the selection vector of a
    MixtureOfExperts container. It also has a persistent alpha and beta, so can be used in complex paths
    to easily sum properly.

    Arguments:

        name (str, optional): Layer name. Defaults to "OutputDeltaBuffer"
    """

    def __init__(self, owns_output=True, owns_delta=True, name=None, alpha=1.0, beta=0.0, bpalpha=1.0, bpbeta=0.0):
        super(OutputDeltaBuffer, self).__init__(name=name)
        self.owns_output = owns_output
        self.owns_delta = owns_delta
        self.alpha = alpha
        self.beta = beta
        self.bpbeta = bpbeta
        self.bpalpha = bpalpha
        self.output_buffer = None

    def __str__(self):
        return "OutputDeltaBuffer : '%s'" % self.name

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(OutputDeltaBuffer, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            in_obj.next_layer = self
        return self

    def fprop(self, inputs, inference=False, alpha=None, beta=None):
        self.outputs[:] = inputs * (self.alpha if alpha is None else alpha) + \
                          self.outputs * (self.beta if beta is None else beta)
        return self.outputs

    def bprop(self, error, alpha=None, beta=None):
        if self.deltas is None:
            return self.deltas
        else:
            self.deltas[:] = error * (self.bpalpha if alpha is None else alpha) + \
                              self.deltas * (self.bpbeta if beta is None else beta)
            return self.deltas
