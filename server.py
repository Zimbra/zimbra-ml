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
from zmlcore.smartfolders.classifier import Config
from neon.util.argparser import NeonArgparser


def main():
    p = NeonArgparser(__doc__)
    options = p.parse_args(gen_be=False)

    Config.options = options

    if Config.options.rng_seed:
        Config.options.rng_seed = int(datetime.datetime.now().timestamp())
    np.random.seed(options.rng_seed)

    print('Server starting on port {}'.format(PORT))
    app = TornadoQL.start(schema=Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier))


if __name__ == '__main__':
    main()
