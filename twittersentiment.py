"""
(C) Copyright 2017, Michael Toutonghi
Author: Michael Toutonghi
Created: 12/24/2017

This example script uses the Zimbra Machine learning GraphQL server to run the Twitter Sentiment competition
on Kaggle. Data is assumed to be in the ./data/twittersentiment/train.csv and ""/test.csv files.
"""
import requests
import configargparse
import pandas as pd
import os
import json
from collections import OrderedDict


class TwitterSentiment:
    classifier_id = '5c8451ad-6aa0-48fe-ab32-558ce3340aec'

    classifier = None
    classifier_fields = 'classifierId vocabPath numSubjectWords numBodyWords numFeatures ' \
                        'exclusiveClasses overlappingClasses epoch trainingSet {date numTrain numTest}'
    training_epochs = 4
    args = None
    classes = OrderedDict([(0, 'positive'), (1, 'negative')])

    @staticmethod
    def instantiate_classifier():
        query = {'query': 'query {classifier(' +
                          'classifierId:{} '.format(json.dumps(ts.classifier_id)) +
                          ') { ' + ts.classifier_fields + ' }}'}
        response = requests.post(ts.args.apiurl.rstrip('/'), json=query)

        if response.ok:
            ts.classifier = json.loads(response.text)['data']['classifier']

        return response.ok

    @staticmethod
    def delete_classifier():
        query = {'query': 'mutation {deleteClassifier(classifierId:"' +
                          '{}") '.format(ts.classifier_id) +
                          '{ result }}'}
        response = requests.post(ts.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_train_set():
        query = {'query': 'mutation {deleteTrainSet(classifierId:"' +
                          '{}") '.format(ts.classifier_id) +
                          '{ result }}'}
        response = requests.post(ts.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def delete_models():
        query = {'query': 'mutation {deleteModels(classifierId:"' +
                          '{}") '.format(ts.classifier_id) +
                          '{ result }}'}
        response = requests.post(ts.args.apiurl.rstrip('/'), json=query)
        return response.ok

    @staticmethod
    def create_classifier():
        mutation = {'query': 'mutation {createClassifier(classifierSpec:{ ' +
                             'classifierId:{} '.format(json.dumps(ts.classifier_id)) +
                             'numSubjectWords:0 numBodyWords:{} numFeatures:0 vocabPath:"{}" '.format(
                                 ts.args.num_words, ts.args.vocab
                             ) +
                             'lookupSize:{} lookupDim:{} '.format(ts.args.lookup_size, ts.args.lookup_dim) +
                             'exclusiveClasses:[{}]'.format(
                                 ' '.join([json.dumps(s) for s in ts.classes.values()])) +
                             ' }) {classifierInfo { ' + ts.classifier_fields + ' }}}'}
        response = requests.post(ts.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            ts.classifier = json.loads(response.text)['data']['createClassifier']['classifierInfo']
            assert ts.classifier_id == ts.classifier['classifierId']

        return response.ok

    @staticmethod
    def train(train, epochs):
        if train is None:
            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(ts.classifier_id) + ' epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(ts.args.learning_rate) + ' }) '
                                 '{ classifierInfo { ' + ts.classifier_fields + ' }}}'}
        else:
            train.set_index(['ItemID'], drop=False, inplace=True)
            train.sort_index(inplace=True)
            exclusive_targets = [ts.classes[int(train['Sentiment'].ix[k])] for k in train.index]

            mutation = {'query': 'mutation {trainClassifier(trainingSpec: { classifierId:' +
                                 json.dumps(ts.classifier_id) +
                                 ' train: {data: [' +
                                 ' '.join(['{url:"' + str(k) + '" text:' +
                                           json.dumps(train['SentimentText'].ix[k]) + '}'
                                           for k in train.index]) +
                                 '] exclusiveTargets: [' +
                                 ' '.join(['"' + s + '"' for s in exclusive_targets]) +
                                 '] } epochs: ' + str(epochs) +
                                 ' learningRate: {}'.format(ts.args.learning_rate) +
                                 ' holdoutPct: 0.1 persist: true }) '
                                 '{ classifierInfo { ' + ts.classifier_fields + ' }}}'}

        response = requests.post(ts.args.apiurl.rstrip('/'), json=mutation)

        if response.ok:
            ts.classifier = json.loads(response.text)['data']['trainClassifier']['classifierInfo']
            assert ts.classifier_id == ts.classifier['classifierId']

        return response.ok

    @staticmethod
    def run():
        if ts.args.delete_models:
            ts.delete_models()

        if ts.args.delete_train_set:
            ts.delete_train_set()

        if ts.args.delete_all:
            ts.delete_classifier()

        # if we have a valid classifier with persisted training data, skip loading it
        if not ts.instantiate_classifier() or ts.classifier['trainingSet'] is None:
            # load the training data
            train = pd.read_csv(os.path.join(ts.args.datapath, 'train.csv'),encoding='latin1')
        else:
            print('Training with persisted training set from {} with {} training and {} test samples'.format(
                ts.classifier['trainingSet']['date'],
                ts.classifier['trainingSet']['numTrain'],
                ts.classifier['trainingSet']['numTest']
            ))
            ts.classes = ts.classifier['exclusiveClasses']
            train = None

        # create the classifier if we don't have one
        if ts.classifier is None:
            if not ts.create_classifier():
                print('Failed to instantiate or create classifier... ', end='')

        if not ts.classifier is None:
            ts.train(train, ts.args.epochs)

        print('finished')


def main():
    p = configargparse.ArgParser(description='This program exercises the Zimbra Machine Learning '
                                             'service while attempting to solve the Twitter Sentiment challenge.')
    p.add_argument('--datapath', type=str, default='./data/twittersentiment',
                   help='Where the twitter sentiment data is located.')
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
    p.add_argument('--vocab', type=str, default='glove.twitter.27B.100d.txt',
                   help='Vocabulary file without directory location to use as word vectors.')
    p.add_argument('--learning_rate', type=float, default=0.001,
                   help='Learning rate for the Adam optimizer.')
    p.add_argument('--lookup_size', type=int, default=0,
                   help='If non-zero, a lookup table and an auto-vocabulary will be used.')
    p.add_argument('--lookup_dim', type=int, default=100,
                   help='Word embedding dimensions when lookup_size specified.')
    p.add_argument('--num_words', type=int, default=60,
                   help='Number of words from each sample to use for scoring.')

    ts.args = p.parse_args()

    # zero lookup table dimensions if no lookup table
    ts.args.lookup_dim = 0 if ts.args.lookup_size == 0 else ts.args.lookup_dim

    print('Kaggle Twitter Sentiment challenge using the Zimbra Machine Learning '
          'classifier. \nCompetition details and datasets can be found at: '
          'https://www.kaggle.com/c/twitter-sentiment-analysis2')

    ts.run()


ts = TwitterSentiment

if __name__ == '__main__':
    main()



