#
# (C) 2017, Michael Toutonghi - All rights reserved.
#
# Author: Michael Toutonghi
# Creation date: 1/12/2017
#
# Miscellaneous custom layers
#
# Licensed as open source under Apache 2.0 license:
# https://www.apache.org/licenses/LICENSE-2.0

# import NN/support libraries from Neon
from neon.layers import Layer
import numpy as np
from numbers import Number


class Noise(Layer):
    """
    A layer to introduce pct_router_noise during fprop. Noise can be controlled by the minimum variables needed, which may
    be added from time to time. Noise is added by a gaussian distribution around each input value with std deviation
    of

    Arguments:

        name (str, optional): Layer name. Defaults to "OutputDeltaBuffer"
    """

    def __init__(self, noise_pct_std=0.05, inference_noise=False, name=None):
        super(Noise, self).__init__(name=name)
        self.noise_pct_std = noise_pct_std
        self.owns_output = True
        self.inference_noise = inference_noise

    def __str__(self):
        return "Noise Layer : '%s' : %.3f%%" % self.name % self.noise_pct_std

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
        super(Noise, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = self.interpret_in_shape(self.in_shape)
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            in_obj.next_layer = self
        return self

    def fprop(self, inputs, inference=False, alpha=None, beta=None):
        # this will be the std dev for the random distribution
        if not inference or self.inference_noise:
            self.outputs[:] = self.be.absolute(inputs * self.noise_pct_std) + 1E-12
            a = self.outputs.get()
            ash = a.shape
            a = np.reshape(a, tuple((a.size, 1)))
            a = np.random.normal(0, a)
            self.outputs[:] = self.be.array(np.reshape(a, ash)) + inputs
        else:
            self.outputs[:] = inputs
        return self.outputs

    def bprop(self, error, alpha=None, beta=None):
        return error

    def interpret_in_shape(self, xshape):
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
                return (np.prod(xshape), 1)

