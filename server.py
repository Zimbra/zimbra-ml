"""
created: 11/8/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

Loads, initializes, and starts the Zimbra machine learning API server.

V1 provides an initiali API for creating, training, and classifying with classifier models

"""
import numpy as np
import datetime
from tornadoql.tornadoql import TornadoQL, PORT
from graphene import Schema
from schema.schema import ClassifierQuery, Mutations, ObserveClassifier
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.util.argparser import extract_valid_args
from zmlcore.neonfixes.transforms import fix_logistic


class Config:
    options = None
    initialized = False

    @staticmethod
    def initialize_neon():
        if not Config.initialized:
            # for now, we don't trust the mkl backend
            if Config.options.backend == 'mkl':
                print('Resetting mkl backend to cpu')
                Config.options.backend = 'cpu'

            be = gen_backend(**extract_valid_args(Config.options, gen_backend))
            # patch a fix to stabilize the CPU version of Neon's logistic function
            fix_logistic(be)
            Config.initialized = True


def main():
    p = NeonArgparser(__doc__)
    options = p.parse_args(gen_be=False)

    Config.options = options

    if Config.options.rng_seed:
        Config.options.rng_seed = int(datetime.datetime.now().timestamp())
    np.random.seed(options.rng_seed)

    Config.initialize_neon()

    print('Server starting on port {}'.format(PORT))
    app = TornadoQL.start(schema=Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier))


if __name__ == '__main__':
    main()
