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
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from zmlcore.neonfixes.transforms import fix_logistic


def main():
    p = NeonArgparser(__doc__)
    p.add_argument('--word_vectors', type=str, required=False, default='./data/glove.6B.100d.txt',
                   help='Path to word vector file, including word, followed by vector per line, space separated')
    options = p.parse_args(gen_be=False)

    if options.rng_seed:
        options.rng_seed = int(datetime.datetime.now().timestamp())
    np.random.seed(options.rng_seed)

    # for now, we don't trust the mkl backend
    if options.backend == 'mkl':
        print('Resetting mkl backend to cpu')
        options.backend = 'cpu'

    be = gen_backend(**extract_valid_args(options, gen_backend))
    # patch a fix to stabilize the CPU version of Neon's logistic function
    fix_logistic(be)

    print('Server starting on port {}'.format(PORT))
    app = TornadoQL.start(schema=Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier))


if __name__ == '__main__':
    main()
