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
from neon.optimizers import Adam
from neon.callbacks.callbacks import Callbacks, TrainMulticostCallback, MetricCallback
from neon.transforms.cost import MultiMetric, Misclassification, LogLoss
from zmlcore.smartfolders.classifier import EmailClassifier
from zmlcore.smartfolders.traincallbacks import TrainingProgress
from zmlcore.data.dataiterator import TrainingIterator
import os, email, mailbox

# random percentage to holdout for validation when training
holdout_pct = 0.1

if __name__ == '__main__':
    p = NeonArgparser(__doc__)
    p.add_argument('--glove', type=str, required=False, default='./data/glove.6B.100d.txt',
                   help='Path to word vector file, including word, followed by vector per line, space separated')
    p.add_argument('--exclusive_classes', type=str, required=False,
                   default='\"finance promos social forums updates\"',
                   help='The labels of the exclusive classes to either train on or classify. ' +
                        'Should be a quoted, space separated list.')
    p.add_argument('--overlapping_classes', type=str, required=False,
                   default='\"important\"',
                   help='The labels of the classes, which can overlap with themselves or an exclusive class. ' +
                        'Should be a quoted, space separated list.')
    p.add_argument('--train', type=bool, required=False, default=False,
                   help='If set to True, the \"--classify\" parameter must either be a maildir with folders that have\n' +
                        'the class names to train present in the names of the folders (ie. \"socialfolder\" ' +
                        'has \"social\" in its name and will be used for training the \"social\" class), ' +
                        'or it should be a path to a training csv file with columns: ' +
                        '\"id, email, overlapping_classes, exclusive_classes\" where each class has its own column.')
    p.add_argument('--receiver_email', type=str, required=False, default='yourname@yourdomain.com',
                   help='An e-mail for training data to consider recipient')
    p.add_argument('--data_path', type=str, required=False, default='./data/mailfolders/', # default='./data/emails.csv',
                   help='Path to either csv file of emails with \"id, email, [overlapping,], [exclusive,]\" columns ' +
                        'or a directory with maildir format mailboxes. If it is a directory, folders with the ' +
                        ' classes specified will be used as sources of test and/or training data, depending on the' +
                        '\"--train\" parameter.')
    p.add_argument('--interim_save_path', type=str, required=False, default=None,
                   help='Specified .csv file to store the email training data, after it has been preprocessed from ' +
                   'maildir format, along with ground truth columns based on finding the classes in the folder names')
    p.add_argument('--results_path', type=str, required=False, default='./data/results.csv',
                   help='The path where a CSV file with email keys and classifications will be written')
    p.add_argument('--learning_rate', type=float, required=False, default=0.001,
                   help='Set learning rate for neural networks')
    options = p.parse_args(gen_be=False)

    random_seed = int(time())
    np.random.seed(random_seed)

    # don't display errors writing to subset copies of DataFrames
    pd.options.mode.chained_assignment = None

    options.batch_size = 1
    be = gen_backend(**extract_valid_args(options, gen_backend))

    optimizer = Adam(learning_rate=options.learning_rate)

    overlapping_classes = options.overlapping_classes.strip(' \"\'').split()
    exclusive_classes = options.exclusive_classes.strip(' \"\'').split()
    classifier = EmailClassifier(options.glove, options.model_file, optimizer=optimizer,
                                 overlapping_classes=overlapping_classes, exclusive_classes=exclusive_classes)

    # determine if we expect to use a csv file or a maildir as our data source
    if os.path.isfile(options.data_path):
        # if file, we expect it to be a .csv file
        emails = pd.DataFrame.from_csv(options.data_path).reset_index()
        if len(emails.columns) == 2 and not options.train:
            emails.columns = ['id', 'message']
        elif len(emails.columns) == (2 + len(overlapping_classes) + len(exclusive_classes)):
            emails.columns = ['id', 'message'] + overlapping_classes + exclusive_classes
        else:
            if options.train:
                print('Training csv requires [id, email, class_columns] columns in the data, one boolean column per ' +
                      'class (0.0 for false, 1.0 for true), overlapping class columns first. Columns must match classes')
            else:
                print('Classification file must either have 2 columns of [id, email] or ' +
                      '[id, email, overlapping, exclusive classes], where the total class columns matches number ' +
                      'of classes')
            emails = None

        # convert to email message objects from strings
        if not emails is None:
            emails['message'] = emails['message'].map(lambda x: email.message_from_string(str(x)))

    elif os.path.isdir(options.data_path):
        # if it's a directory, it should be a directory of maildir folders from isync
        folders = [f.strip('. ') for f in next(os.walk(options.data_path))[1]]

        if options.train:
            overlapping_folders = {c: [] for c in overlapping_classes}
            exclusive_folders = {c: [] for c in exclusive_classes}
            for f in folders:
                for c in exclusive_classes:
                    if c in f:
                        exclusive_folders[c].append(f)
                for c in overlapping_classes:
                    if c in f:
                        overlapping_folders[c].append(f)

            # remove zero length lists of classes
            overlapping_folders = {k: v for k, v in overlapping_folders.items() if len(v) > 0}
            exclusive_folders = {k: v for k, v in exclusive_folders.items() if len(v) > 0}

            if len(overlapping_folders) + len(exclusive_folders) == len(overlapping_classes) + len(exclusive_classes):
                # now, we have a list of folders for each overlapping class and each exclusive class
                # important, automated, financial messages are likely in a folder called importantautomatedfinancial
                # while personal, unimportant, shopping messages would likely be in shopping. in any case,
                # we may have common folder names in multiple classes, and if so, we should not duplicate loading
                # of messages, but instead, add classes to the message id, as the ids will match
                loaded_folders = {}

                # remove duplicate folder names
                for l in list(overlapping_folders.values()) + list(exclusive_folders.values()):
                    for k in l:
                        loaded_folders[k] = None

                # dict of {msgid: message} for each folder
                maildir = mailbox.Maildir(options.data_path)
                loaded_folders = {f: {k: v for k, v in maildir.get_folder(f).items()}
                                  for f in loaded_folders}

                num_emails = sum([len(d) for d in loaded_folders.values()])

                # make large, empty dataframe for all messages
                emails = pd.DataFrame(index=pd.Index(data=range(num_emails)),
                                      columns=['id', 'message'] + overlapping_classes + exclusive_classes)

                idx = 0
                # populate the dataframe
                for k, v in loaded_folders.items():
                    # determine the class membership for all messages in this folder
                    targets = [1.0 if t in k else 0.0 for t in overlapping_classes] + \
                              [1.0 if t in k else 0.0 for t in exclusive_classes]

                    for id, m in v.items():
                        emails.iloc[idx] = pd.Series(data=[id, m] + targets, index=emails.columns)
                        idx += 1

                if options.interim_save_path:
                    pd.DataFrame.to_csv(options.interim_save_path, index=False)
            else:
                print('Training with a mailbox directory requires mailbox folders to contain the class names to train')
                emails = None
        else:
            # load mail from inbox and classify
            raise NotImplemented()
    else:
        print('No data source file or maildir directory available. {} not found.'.format(options.data_path))
        emails = None

    # we now have a dataframe, emails, that contains either the emails to classify or the emails to
    # train and test from
    if options.train:
        # if train and test, we need to shuffle and split the data, prepare it for
        # training, fit to the neural network, then save the network model

        # shuffle
        valid_df = emails.sample(frac=holdout_pct)
        train_df = emails.drop(valid_df.index).sample(frac=1)

        ol_len = len(overlapping_classes)

        valid = TrainingIterator(classifier.emails_to_nn_representation(list(valid_df['message'].values),
                                                                        receiver_address=options.receiver_email),
                                 [[be.array(a[:ol_len]),
                                   be.array(a[ol_len:])]
                                  for a in valid_df.loc[:, overlapping_classes + exclusive_classes].values])
        train = TrainingIterator(classifier.emails_to_nn_representation(list(train_df['message'].values),
                                                                        receiver_address=options.receiver_email),
                                 [[be.array(a[:ol_len]),
                                   be.array(a[ol_len:])]
                                  for a in train_df.loc[:, overlapping_classes + exclusive_classes].values])

        callbacks = Callbacks(classifier.neuralnet, **options.callback_args)
        callbacks.add_callback(TrainingProgress(valid))
        print('Training neural networks on {} samples for {} epochs'.format(len(train_df), options.epochs))
        classifier.neuralnet.eval(valid, MultiMetric(Misclassification(), 1))
        classifier.fit(train, optimizer, options.epochs, callbacks)
    else:
        # if we are to classify, then we need to create dataframe with the classes and save it to our results path
        df = emails

        print('Timing neural network conversion and classification of {} emails'.format(np.minimum(1000,len(df))))
        start = pd.datetime.utcnow()
        batch = []
        for i in range(0, np.minimum(1000, len(df)), options.batch_size):
            batch += classifier.classify(list(df.iloc[i:i + options.batch_size]['message'].values))
        finish = pd.datetime.utcnow()
        print('all emails processed and classified, took {} to complete'.format(finish-start))
    exit(0)
