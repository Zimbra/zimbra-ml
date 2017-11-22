"""
created: 11/1/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

GraphQL schema for the Zimbra email and text classification micro-service.

The general idea is that

"""
import graphene
import urllib3
from zmlcore.smartfolders.classifier import EmailClassifier
import uuid


DEFAULT_MODEL_PATH = 'data/default_model.pkl'
DEFAULT_VOCAB_PATH = 'data/glove.6B.100d.txt'
DEFAULT_OVERLAPPING = ['important']
DEFAULT_EXCLUSIVE = ['finance', 'promos', 'social', 'forums', 'updates']
DEFAULT_NUM_REQUESTS = 10
DEFAULT_CLASSIFIER_ID = str(uuid.uuid4())
http = urllib3.PoolManager(num_pools=DEFAULT_NUM_REQUESTS)
_global_classifiers = {}


def get_content(path):
    r = http.request('GET', path)
    return r.read().decode(r.info().get_content_charset())


class OverlappingClassification(graphene.ObjectType):
    class Meta:
        description = 'Classifications for overlapping classes are not scalars, as they include a probability of ' \
                      'class membership.'
    name = graphene.String(required=True)
    probability = graphene.Float(required=True)

class TextClasses(graphene.ObjectType):
    class Meta:
        description = 'Classifications returned for classifying text or email'

    url = graphene.String(description='Always returned - used to identify the classified text.',
                          required=True)
    exclusive = graphene.String(description='Exclusive classification. Is None if classification fails.',
                                default_value=None)
    overlapping = graphene.List(graphene.String, description='List of overlapping classifications.', default_value=None)


class TextFeatures(graphene.InputObjectType):
    class Meta:
        description = 'Additional features that can be any input data in addition to the content itself to be used ' \
                      'in the model.'
    features = graphene.List(graphene.Float,
                             default_value=None,
                             description='Additional features that should be the same for training and classification, '
                                         'which enable support for features to be considered as well as content.')


class MessageFeatures(TextFeatures):
    class Meta:
        description = 'Additional features over those available in TextFeatures that are specific to email messages. '
    receiver_email = graphene.String(description='Classify with respect to this receiver. If this is present, a very '
                                                 'rudimentary, English-only check of the headers from the perspective '
                                                 'of that receiver generates features used for classification. '
                                                 'Current features include:\n'
                                                 '1. Is the mail to the receiver? 0/1\n'
                                                 '2. Is the mail exclusively to the receiver? 0/1\n'
                                                 '3. Is it a reply with the receiver on the "to" line? 0/1\n'
                                                 '4. Is it a forward? 0/1\n', default_value=None)


class Text(graphene.InputObjectType):
    class Meta:
        description = 'Parameter representing a text document. Either the url should be a valid link to the ' \
                      'text to be classified in text/html or text/plain format, or the text field itself must ' \
                      'contain the content.'
    url = graphene.String(description='URL of the text, which can be either a file or http location. The contents, '
                                      'if it is a file or for the "text" field, must be identified by the "mime" '
                                      'field. If the "text" field is not None, it is assumed to be the content '
                                      'referred to by this URL, and no request is made of the URL. Both text and '
                                      'email payloads can be delivered this way. If text is not none, URL can be '
                                      'an identifying URI. This field is required, and is used as the ID for the '
                                      'classification on return.',
                          required=True)
    mime = graphene.String(description='Mime type of the payload.',
                           default_value='text/plain')
    text = graphene.String(description='Optional parameter that, if present, eliminates the need to retrieve '
                                       'the text from the URL field.',
                           default_value=None)
    text_features = TextFeatures(description='Optional parameter for classifiers for training with features in '
                                             'addition to the content.',
                                 default_value=None)


class EmailMessage(Text):
    class Meta:
        description = 'Subclass of Text, where text field or document linked is in email format and features are ' \
                      'message features.'


class TrainingData(graphene.InputObjectType):
    class Meta:
        description = 'Training data and targets to train a classifier.'
    data = graphene.List(EmailMessage, description='List of text document samples or email messages for training.')
    exclusive_targets = graphene.List(graphene.String,
                                      description='A list of exclusive classes that correspond '
                                                  'positionally to the list of messages.',
                                      default_value=None)
    overlapping_targets = graphene.List(graphene.List(graphene.String),
                                      description='A list of lists of overlapping targets, which can overlap with '
                                                  'each other or with exclusive targets. Overlapping targets are '
                                                  'represented by a probability.',
                                      default_value=None)


class ClassifierSpec(graphene.InputObjectType):
    class Meta:
        description = 'Everything needed to identify or instantiate a classifier for inference or training.' \
                      'If ID is present, we will attempt to use the cached classifier in all cases.'
    classifier_id = graphene.String(description='UUID, which if present, allows the server to use a cached '
                                                'classifier without reloading. This ID is returned from any '
                                                'API that may instantiate a classifier. It is optional for '
                                                'instantiation of new classifiers, as a default UUID is provided.',
                                    default_value=DEFAULT_CLASSIFIER_ID)
    vocab_path = graphene.String(description='Optional path to an alternate vocabulary file. This API currently '
                                             'supports the Stanford GloVe format files.',
                                 default_value=DEFAULT_VOCAB_PATH)
    model_path = graphene.String(description='Optional path to a pre-trained model to load for this classifier ',
                                 default_value=DEFAULT_MODEL_PATH)
    num_subject_words = graphene.Int(description='Number of subject words. Currently, subject and body words must add '
                                                 'up to either 30 or 60. The total is used for text.',
                                     default_value=8)
    num_body_words = graphene.Int(description='Number of body words to extract from the text or email.',
                                  default_value=52)
    num_features = graphene.Int(description='Number of additional features, over and above content that will '
                                            'be considered in the classifier. The default of 4 assumes processing of '
                                            'receiver e-mail, and should not be used if that is not the desired '
                                            'behavior.',
                                default_value=4)
    exclusive_classes = graphene.List(graphene.String,
                                      description='A list of strings representing the exclusive classes to be '
                                                  'trained or considered in a trained classifier.',
                                      default_value=DEFAULT_EXCLUSIVE)
    overlapping_classes = graphene.List(graphene.String,
                                        description='A list of strings representing the overlapping classes to be '
                                                    'trained or considered in a trained classifier.',
                                        default_value=DEFAULT_OVERLAPPING)


class ClassifierInfo(graphene.ObjectType):
    class Meta:
        description = 'Return information about an instantiated classifier, including ID and epoch.'
    classifier_id = graphene.String(description='UUID that allows the server to use a cached '
                                                'classifier without reloading. This should be passed in the '
                                                'ClassifierSpec of subsequent calls. Common UUIDs can be used '
                                                'across servers to identify shared classifiers.')
    vocab_path = graphene.String(description='Path to the vocabulary used.'),
    model_path = graphene.String(description='Path to the model for this classifier, which can be reused '
                                             'after training for subsequent classification and additional '
                                             'training.'),
    num_subject_words = graphene.Int(description='Current number of subject words of the classifer.'),
    num_body_words = graphene.Int(description='Current number of body words of the classifer.'),
    num_analytics_features = graphene.Int(description='Number of features, in addition to content, considered.'),
    exclusive_classes = graphene.List(graphene.String, description='A list of the exclusive classes handled by this '
                                                                   'classifier.')
    overlapping_classes = graphene.List(graphene.String,
                                        description='A list of strings representing the overlapping classes to be '
                                                    'trained or considered in a trained classifier.')
    epoch = graphene.Int(description='The number of separate epochs of training this classifer has already had.',
                         default_value=0)


class TrainingSpec(graphene.InputObjectType):
    class Meta:
        description = 'Everything needed to identify and train a classifier, including number of epochs,' \
                      'optimizer, and learning rate.'
    classifier_spec = ClassifierSpec()
    data = TrainingData()
    epochs = graphene.Int(description='Number of epochs to train with the data passed. If the classifier is already '
                                      'trained, this will add epochs to its training', default_value=5)
    optimizer = graphene.String(description='The supported optimization algorithm to use for neural networks.',
                                default_value='adam')
    learning_rate = graphene.String(description='The learning rate for training.', default_value=0.001)


class ClassifierQuery(graphene.ObjectType):
    class Meta:
        description = 'The main query for classifying text and emails'

    classifier_info = graphene.Field(ClassifierInfo, description='The instantiated classifier information.',
                                     classifier_spec=ClassifierSpec())

    classifications = graphene.List(TextClasses,
                                    description='Classifications of data passed.',
                                    classifier_spec=ClassifierSpec(),
                                    data=graphene.List(Text))

    @staticmethod
    def resolve_classifier_info(root, info, classifier_spec):
        c_info = _global_classifiers.get(classifier_spec.classifier_id, None)
        if c_info is None:
            classifier = EmailClassifier(classifier_spec.vocab_path,
                                         classifier_spec.model_path,
                                         exclusive_classes=classifier_spec.exclusive_classes,
                                         overlapping_classes=classifier_spec.overlapping_classes,
                                         num_analytics_features=classifier_spec.num_analytics_features,
                                         num_subject_words=classifier_spec.num_subject_words,
                                         num_body_words=classifier_spec.num_body_words)

            # get and ID and store in our local dict as a cache
            _global_classifiers[classifier_spec.classifier_id] = classifier

            c_info = ClassifierInfo(classifier_id = classifier_spec.classifier_id,
                                    vocab_path = classifier_spec.vocab_path,
                                    model_path = classifier_spec.model_path,
                                    num_subject_words = classifier_spec.graphene.Int(),
                                    num_body_words = classifier_spec.graphene.Int(),
                                    num_analytics_features = classifier_spec.graphene.Int(),
                                    exclusive_classes = classifier_spec.graphene.List(graphene.String),
                                    overlapping_classes = classifier_spec.graphene.List(graphene.String),
                                    epoch = classifier.neuralnet.epoch_index)

        return c_info

    @staticmethod
    def resolve_classifications(root, info, classifier_spec, data):
        classifier = _global_classifiers.get(ClassifierQuery.resolve_classifier_info(None,
                                                                                     None,
                                                                                     classifier_spec).classifier_id,
                                             None)
        if not classifier is None and len(data) > 0:
            if isinstance(data[0], EmailMessage):
                # TODO: breakout features per text
                # classify email messages
                emails = [em.text for em in data]
                class_vectors = classifier.classify(emails, inference=True)
            else:
                # TODO: classify just text
                pass

            # TODO: format results as classes, not the float arrays returned from the classifier

        return [TextClasses(id=tl['uri'], overlapping=['overlapping'], exclusive=['exclusive']) for tl in data]


class TrainClassifier(graphene.Mutation):
    class Meta:
        description = 'This mutation is used to train an instantiated classifier.'
    class Arguments:
        train = TrainingSpec()

    classifier_info = graphene.Field(ClassifierInfo)

    def mutate(root, info, train):
        classifier_info = ClassifierQuery.resolve_classifier_info(None, None, train.classifier_spec)
        # TODO: train the classifier for the specified epochs
        return classifier_info


class Mutations(graphene.ObjectType):
    train_classifier = TrainClassifier.Field()


class ObserveClassifier(graphene.ObjectType):
    class Meta:
        description = 'Provides the ability to subscribe to real-time classifier training data streams.'

    place_holder = graphene.Int()

    def resolve_place_holder(root, info):
        pass


def main():
    schema = graphene.Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier)
    result = schema.execute('{classifications(classifierSpec: {}, data: [{url:"test"}]) {url, exclusive}}')
    x = result.data
    print(x)

if __name__ == '__main__':
    main()