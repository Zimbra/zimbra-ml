"""
created: 11/8/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

Loads, initializes, and starts the Zimbra machine learning API server.

V1 provides an initiali API for creating, training, and classifying with classifier models

"""
from flask import Flask
from flask_graphql import GraphQLView
from graphene import Schema
from schema.schema import ClassifierQuery

HOST = 'localhost'
PORT = 5000

def create_app(path='/graphql', **kwargs):
    app = Flask(__name__)

    app.add_url_rule('/graphql',
                     view_func=GraphQLView.as_view('graphql', schema=Schema(query=ClassifierQuery), **kwargs))
    return app


def main():
    app = create_app(graphiql=True)
    app.run(host=HOST, port=PORT)

if __name__ == '__main__':
    main()
