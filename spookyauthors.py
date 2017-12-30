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
    classifier = None
    classifier_fields = 'classifierId vocabPath numSubjectWords numBodyWords numFeatures ' \
                        'exclusiveClasses overlappingClasses epoch trainingSet {date numTrain numTest}'
    training_epochs = 4
    args = None
    classes = None

    @staticmethod
    def instantiate_classifier():
        query = {'query': 'query {classifier(' +
                          'classifierId:{} '.format(json.dumps(sa.classifier_id)) +
                          ') { ' + sa.classifier_fields + ' }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)

        if response.ok:
            sa.classifier = json.loads(response.text)['data']['classifier']

        return response.ok

    @staticmethod
    def delete_classifier():
        query = {'query': 'mutation {deleteClassifier(classifierId:"' +
                          '{}") '.format(sa.classifier_id) +
                          '{ result }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_train_set():
        query = {'query': 'mutation {deleteTrainSet(classifierId:"' +
                          '{}") '.format(sa.classifier_id) +
                          '{ result }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_models():
        query = {'query': 'mutation {deleteModels(classifierId:"' +
                          '{}") '.format(sa.classifier_id) +
                          '{ result }}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def create_classifier():
        mutation = {'query': 'mutation {createClassifier(classifierSpec:{ ' +
                             'classifierId:{} '.format(json.dumps(sa.classifier_id)) +
                             'numSubjectWords:0 numBodyWords:{} numFeatures:0 vocabPath:"{}" '.format(
                                 sa.args.num_words, sa.args.vocab
                             ) +
                             'lookupSize:{} lookupDim:{} '.format(sa.args.lookup_size, sa.args.lookup_dim) +
                             'exclusiveClasses:[{}]'.format(
                                 ' '.join([json.dumps(s) for s in sa.classes])) +
                             ' }) {classifierInfo { ' + sa.classifier_fields + ' }}}'}
        response = requests.post(sa.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            sa.classifier = json.loads(response.text)['data']['createClassifier']['classifierInfo']
            assert sa.classifier_id == sa.classifier['classifierId']

        return response.ok

    @staticmethod
    def train(train, epochs):
        if train is None:
            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(sa.classifier_id) + ' epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(sa.args.learning_rate) + ' }) '
                                 '{ classifierInfo { ' + sa.classifier_fields + ' }}}'}
        else:
            train.set_index(['id'], drop=False, inplace=True)
            train.sort_index(inplace=True)
            exclusive_targets = [train['author'].ix[k] for k in train.index]

            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(sa.classifier_id) +
                                 ' train: {data: [' +
                                 ' '.join(['{url:"' + train['id'].ix[k] + '" text:' +
                                           json.dumps(train['text'].ix[k]) + '}'
                                           for k in train.index]) +
                                 '] exclusiveTargets: [' +
                                 ' '.join(['"' + s + '"' for s in exclusive_targets]) +
                                 '] } epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(sa.args.learning_rate) +
                                 ' holdoutPct: 0.1 persist: true }) '
                                 '{ classifierInfo { ' + sa.classifier_fields + ' }}}'}

        response = requests.post(sa.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            sa.classifier = json.loads(response.text)['data']['trainClassifier']['classifierInfo']
            assert sa.classifier_id == sa.classifier['classifierId']

        return response.ok

    @staticmethod
    def run():
        if sa.args.delete_models:
            sa.delete_models()

        if sa.args.delete_train_set:
            sa.delete_train_set()

        if sa.args.delete_all:
            sa.delete_classifier()

        # if we have a valid classifier with persisted training data, skip loading it
        if not sa.instantiate_classifier() or sa.classifier['trainingSet'] is None:
            # load the training data
            train = pd.read_csv(os.path.join(sa.args.datapath, 'train.csv'))

            # DEBUG
            #train = train.sample(n=10)

            # determine the classes we will use
            classes = train.set_index([train.columns[-1]], drop=False)
            sa.classes = [v for v in classes[~classes.index.duplicated(keep='first')][train.columns[-1]].values]
        else:
            print('Training with persisted training set from {} with {} training and {} test samples'.format(
                sa.classifier['trainingSet']['date'],
                sa.classifier['trainingSet']['numTrain'],
                sa.classifier['trainingSet']['numTest']
            ))
            sa.classes = sa.classifier['exclusiveClasses']
            train = None

        # create the classifier if we don't have one
        if sa.classifier is None:
            if not sa.create_classifier():
                print('Failed to instantiate or create classifier... ', end='')

        if not sa.classifier is None:
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
    p.add_argument('--delete_models', type=bool, default=False,
                   help='Delete all trained models for the classifier before training while leaving training set'
                        'and model intact.')
    p.add_argument('--delete_train_set', type=bool, default=False,
                   help='Delete the training set for a classifier and create a new one to retrain.')
    p.add_argument('--delete_all', type=bool, default=False,
                   help='Delete the classifier entirely and create a new one.')
    p.add_argument('--epochs', type=int, default=5,
                   help='Total number of epochs the model should be trained, including those already done.')
    p.add_argument('--vocab', type=str, default='glove.6B.100d.txt',
                   help='Vocabulary file without directory location to use as word vectors.')
    p.add_argument('--learning_rate', type=float, default=0.001,
                   help='Learning rate for the Adam optimizer.')
    p.add_argument('--lookup_size', type=int, default=0,
                   help='If non-zero, a lookup table and an auto-vocabulary will be used.')
    p.add_argument('--lookup_dim', type=int, default=100,
                   help='Word embedding dimensions when lookup_size specified.')
    p.add_argument('--num_words', type=int, default=60,
                   help='Number of words from each sample to use for scoring.')

    sa.args = p.parse_args()

    # zero lookup table dimensions if no lookup table
    sa.args.lookup_dim = 0 if sa.args.lookup_size == 0 else sa.args.lookup_dim

    print('Kaggle Spooky Author Identification challenge using the Zimbra Machine Learning '
          'classifier. \nCompetition details and datasets can be found at: '
          'https://www.kaggle.com/c/spooky-author-identification')

    sa.run()


sa = SpookyAuthors

if __name__ == '__main__':
    main()



