"""
(C) Copyright 2017, Michael Toutonghi
Author: Michael Toutonghi
Created: 12/24/2017

This is a test of the overlapping class classifier, which uses the Zimbra Machine learning GraphQL server to
run the Jigsaw toxic challenge competition on Kaggle.
Data is assumed to be in the ./data/toxicchallenge/train.csv and ""/test.csv files.

"""
import requests
import configargparse
import pandas as pd
import os
import json
from collections import OrderedDict

_DEBUG = True
_DEBUG_SAMPLE_SIZE = 50000

class ToxicChallenge:
    classifier_id = '4c8451ad-6aa0-48fe-ab32-558ce3340aed'

    classifier = None
    classifier_fields = 'classifierId vocabPath numSubjectWords numBodyWords numFeatures ' \
                        'exclusiveClasses overlappingClasses epoch trainingSet {date numTrain numTest}'
    training_epochs = 4
    args = None
    classes = None

    @staticmethod
    def instantiate_classifier():
        query = {'query': 'query {classifier(' +
                          'classifierId:{} '.format(json.dumps(tc.classifier_id)) +
                          ') { ' + tc.classifier_fields + ' }}'}
        response = requests.post(tc.args.apiurl.rstrip('/'), json=query)

        if response.ok:
            tc.classifier = json.loads(response.text)['data']['classifier']

        return response.ok

    @staticmethod
    def delete_classifier():
        query = {'query': 'mutation {deleteClassifier(classifierId:"' +
                          '{}") '.format(tc.classifier_id) +
                          '{ result }}'}
        response = requests.post(tc.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_train_set():
        query = {'query': 'mutation {deleteTrainSet(classifierId:"' +
                          '{}") '.format(tc.classifier_id) +
                          '{ result }}'}
        response = requests.post(tc.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_models():
        query = {'query': 'mutation {deleteModels(classifierId:"' +
                          '{}") '.format(tc.classifier_id) +
                          '{ result }}'}
        response = requests.post(tc.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def create_classifier():
        mutation = {'query': 'mutation {createClassifier(classifierSpec:{ ' +
                             'classifierId:{} '.format(json.dumps(tc.classifier_id)) +
                             'numSubjectWords:0 numBodyWords:{} numFeatures:0 vocabPath:"{}" '.format(
                                 tc.args.num_words, tc.args.vocab
                             ) +
                             'lookupSize:{} lookupDim:{} '.format(tc.args.lookup_size, tc.args.lookup_dim) +
                             'overlappingClasses:[{}]'.format(
                                 ' '.join([json.dumps(s) for s in tc.classes])) +
                             ' }) {classifierInfo { ' + tc.classifier_fields + ' }}}'}
        response = requests.post(tc.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            tc.classifier = json.loads(response.text)['data']['createClassifier']['classifierInfo']
            assert tc.classifier_id == tc.classifier['classifierId']

        return response.ok

    @staticmethod
    def train(train, epochs):
        if train is None:
            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(tc.classifier_id) + ' epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(tc.args.learning_rate) + ' }) '
                                 '{ classifierInfo { ' + tc.classifier_fields + ' }}}'}
        else:
            train.set_index(['id'], drop=False, inplace=True)
            train.sort_index(inplace=True)
            overlapping_targets = [[c for c in train.columns[2:] if train[c].ix[k] != 0] for k in train.index]

            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(tc.classifier_id) +
                                 ' train: {data: [' +
                                 ' '.join(['{url:"' + str(k) + '" text:' +
                                           json.dumps(train['comment_text'].ix[k]) + '}'
                                           for k in train.index]) +
                                 '] overlappingTargets: [' +
                                 ' '.join(['[' + ' '.join(['"' + s + '"' for s in l]) + ']' for l in overlapping_targets]) +
                                 '] } epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(tc.args.learning_rate) +
                                 ' holdoutPct: 0.1 persist: {} '.format('true') + '}) ' +
                                 '{ classifierInfo { ' + tc.classifier_fields + ' }}}'}

        response = requests.post(tc.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            tc.classifier = json.loads(response.text)['data']['trainClassifier']['classifierInfo']
            assert tc.classifier_id == tc.classifier['classifierId']

        return response.ok

    @staticmethod
    def run():
        if tc.args.delete_models:
            tc.delete_models()

        if tc.args.delete_train_set:
            tc.delete_train_set()

        if tc.args.delete_all:
            tc.delete_classifier()

        # if we have a valid classifier with persisted training data, skip loading it
        if not tc.instantiate_classifier() or tc.classifier['trainingSet'] is None:
            # load the training data
            train = pd.read_csv(os.path.join(tc.args.datapath, 'train.csv'), encoding='iso-8859-1')
            tc.classes = train.columns[2:]

            if _DEBUG:
                train = train.sample(n=_DEBUG_SAMPLE_SIZE)
        else:
            print('Training with persisted training set from {} with {} training and {} test samples'.format(
                tc.classifier['trainingSet']['date'],
                tc.classifier['trainingSet']['numTrain'],
                tc.classifier['trainingSet']['numTest']
            ))
            tc.classes = tc.classifier['overlappingClasses']
            train = None

        # create the classifier if we don't have one
        if tc.classifier is None:
            if not tc.create_classifier():
                print('Failed to instantiate or create classifier... ', end='')

        if tc.classifier is not None:
            tc.train(train, tc.args.epochs)

        print('finished')


def main():
    p = configargparse.ArgParser(description='This program exercises the Zimbra Machine Learning '
                                             'service while attempting to solve the toxic challenge.')
    p.add_argument('--datapath', type=str, default='./data/toxicchallenge',
                   help='Where the toxic challenge data is located.')
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

    tc.args = p.parse_args()

    # zero lookup table dimensions if no lookup table
    tc.args.lookup_dim = 0 if tc.args.lookup_size == 0 else tc.args.lookup_dim

    print('Kaggle Toxic Comment Classification challenge using the Zimbra Machine Learning '
          'classifier. \nCompetition details and datasets can be found at: '
          'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data')

    tc.run()


tc = ToxicChallenge

if __name__ == '__main__':
    main()



