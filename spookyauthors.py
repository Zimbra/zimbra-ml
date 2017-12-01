"""
(C) Copyright 2017, Michael Toutonghi
Author: Michael Toutonghi
Created: 11/28/2017

This example script uses the Zimbra Machine learning GraphQL server to run the Spooky Author Identification competition
on Kaggle. Data is assumed to be in the ./data/spookyauthor/train.csv and ""/test.csv files.
"""
import requests
import configargparse
import pandas as pd
import os
import json


class SpookyAuthors:
    classifier_id = 'fef28da2-0d26-4e68-a383-6ad195910de4'
    training_epochs = 8
    epoch = 0
    args = None
    classes = None

    @staticmethod
    def instantiate_classifier():
        query = {'query': 'query {classifier(' +
                          'classifierId:{} '.format('"' + sa.classifier_id + '"') +
                          ') { classifierId vocabPath numSubjectWords numBodyWords numFeatures '
                          'exclusiveClasses overlappingClasses epoch }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)

        if response.ok:
            sa.epoch = json.loads(response.text)['data']['classifier']['epoch']

        return response.ok

    @staticmethod
    def delete_classifier():
        query = {'query': 'mutation {deleteClassifier(classifierId:"' +
                          '{}") '.format(sa.classifier_id) +
                          '{ result }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def create_classifier():
        mutation = {'query': 'mutation {createClassifier(classifierSpec:{ ' +
                             'classifierId:{} '.format(json.dumps(sa.classifier_id)) +
                             'numSubjectWords:0 numBodyWords:60 numFeatures:0 ' +
                             'exclusiveClasses:[{}]'.format(
                                 ' '.join([json.dumps(s) for s in sa.classes])) +
                             ' }) {classifierInfo { classifierId }}}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            assert sa.classifier_id == \
                   json.loads(response.text)['data']['createClassifier']['classifierInfo']['classifierId']

        return response.ok

    @staticmethod
    def train(train, epochs):
        train.set_index(['id'], drop=False, inplace=True)
        train.sort_index(inplace=True)
        exclusive_targets = [train['author'].ix[k] for k in train.index]

        mutation = {'query': 'mutation {trainClassifier(spec: { classifierId: "' + sa.classifier_id + '" ' +
                             'train: {data: [' +
                             ' '.join(['{url:"' + train['id'].ix[k] + '" text:' + json.dumps(train['text'].ix[k]) + '}'
                                       for k in train.index]) +
                             '] exclusiveTargets: [' +
                             ' '.join(['"' + s + '"' for s in exclusive_targets]) +
                             '] } epochs: ' + str(epochs) +
                             ' holdoutPct: 0.1}) { classifierInfo { classifierId exclusiveClasses epoch }}}'}

        response = requests.post(sa.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            assert sa.classifier_id == \
                   json.loads(response.text)['data']['trainClassifier']['classifierInfo']['classifierId']

        return response.ok

    @staticmethod
    def run():
        if sa.args.delete:
            sa.delete_classifier()

        # load the training data
        train = pd.read_csv(os.path.join(sa.args.datapath, 'train.csv'))

        # determine the classes we will use
        classes = train.set_index([train.columns[-1]], drop=False)
        sa.classes = [v for v in classes[~classes.index.duplicated(keep='first')][train.columns[-1]].values]

        # instantiate or create the classifier
        if sa.instantiate_classifier():
            training_epochs = max(0, sa.args.epochs - sa.epoch)
        else:
            if sa.create_classifier():
                training_epochs = sa.args.epochs
            else:
                print('Failed to instantiate or create classifier... ', end='')
                training_epochs = 0

        if training_epochs:
            # prepare training data
            sa.train(train, sa.args.epochs)


        print('finished')


def main():
    sa = SpookyAuthors
    p = configargparse.ArgParser(description='This program exercises the Zimbra Machine Learning '
                                             'service while attempting to solve the Kaggle Spooky Author '
                                             'identification challenge.')
    p.add_argument('--datapath', type=str, default='./data/spookyauthors',
                   help='Where the spooky author data is located.')
    p.add_argument('--apiurl', type=str, default='http://localhost:8888/graphql',
                   help='The machine learning API endpoint.')
    p.add_argument('--delete', type=bool, default=False,
                   help='Delete the classifier and create a new one to retrain.')
    p.add_argument('--epochs', type=int, default=5,
                   help='Total number of epochs the model should be trained, including those already done.')

    sa.args = p.parse_args()

    print('Kaggle Spooky Author Identification challenge, using the Zimbra Machine Learning '
          'classifier. \nCompetition details and datasets can be found at: '
          'https://www.kaggle.com/c/spooky-author-identification')

    sa.run()


sa = SpookyAuthors

if __name__ == '__main__':
    main()



