"""
created: 9/8/2017
(c) copyright 2017 Synacor, Inc

Loads, initializes, and runs the Zimbra machine learning server.

V1 provides smart folder tag categorization and contact suggestion lists.

"""
import numpy as np
import pandas as pd
from time import time
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from .smartfolders import EmailClassifier
import email

overlapping_classes=['important', 'automated']
exclusive_classes=['financial', 'shopping', 'social', 'travel', 'business']

if __name__ == '__main__':
    p = NeonArgparser(__doc__)
    p.add_argument('--glove', required=False, default='./data/glove.6B.100d.txt',
                   help='path to GloVe file, including word, followed by vector per line, space separated')
    p.add_argument('--train', required=False, default=None,
                   help='path to training csv file with \"id, email, {}, {}\"'.format(
        overlapping_classes, exclusive_classes))
    p.add_argument('--classify', required=False, default='./data/emails.csv',
                   help='path to csv emails with \"id, email\" in the first two columns')
    options = p.parse_args()

    random_seed = int(time())
    np.random.seed(random_seed)

    # don't display errors writing to subset copies of DataFrames
    pd.options.mode.chained_assignment = None

    options.batch_size = 1
    be = gen_backend(**extract_valid_args(options, gen_backend))

    classifier = EmailClassifier(options.glove, options.model_file)

    # we are either training or classifying, for now, classifying is through a file, in the future,
    # we will do it through the milter API
    if not options.train is None:
        print('not implemented')
    else:
        # load the classification file, convert to nn_input and loop through to classify
        emails = []
        df = pd.DataFrame.from_csv(options.classify)

        print('Timing neural network conversion and classification of {} emails'.format(np.minimum(1000,len(df))))
        start = pd.datetime.utcnow()
        for i in range(0, np.minimum(1000,len(df)), options.batch_size):
            batch = classifier.classify(list(df.iloc[i:i+options.batch_size]['message'].map(
                                                             lambda x: email.message_from_string(str(x))).values))
        finish = pd.datetime.utcnow()
        print('all emails processed and classified, took {} to complete'.format(finish-start))
    exit(0)