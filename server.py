"""
created: 11/8/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

Loads, initializes, and starts the Zimbra machine learning API server.

V1 provides an initiali API for creating, training, and classifying with classifier models

"""
from tornadoql.tornadoql import TornadoQL, PORT
from graphene import Schema
from schema.schema import ClassifierQuery, Mutations, ObserveClassifier

def main():
    print('Server starting on port {}'.format(PORT))
    app = TornadoQL.start(schema=Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier))

if __name__ == '__main__':
    main()
