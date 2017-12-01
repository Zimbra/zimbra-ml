"""
created: 11/1/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

GraphQL schema for the Zimbra email and text classification micro-service.

The general idea is that

"""
import os
import graphene
import urllib3
import numpy as np
from zmlcore.smartfolders.classifier import EmailClassifier
import uuid
import email
import json
from concurrent.futures import ThreadPoolExecutor


class DEFAULTS:
    MODEL_PATH = 'data/models/'
    MODEL_EXT = '.model'
    VOCAB_PATH = 'data/vocabularies/'
    VOCAB_FILE = 'glove.6B.100d.txt'
    META_PATH = 'data/meta/'
    TRAIN_PATH = 'data/meta/'
    META_EXT = '.zml'
    TRAIN_EXT = '.train'
    OVERLAPPING = None
    EXCLUSIVE = ['finance', 'promos', 'social', 'forums', 'updates']
    NUM_REQUESTS = 10

http = urllib3.PoolManager(num_pools=DEFAULTS.NUM_REQUESTS)
_global_classifier_threads = ThreadPoolExecutor(1) # only one thread, as the ML models are not thread safe
_global_classifiers = {}
_global_classifier_info = {}


def get_content_as_bytes(path):
    if not os.path.isfile(path):
        r = http.request('GET', path)
        return r.read()
    else:
        with open(path, 'r') as f:
            return f.read()


def get_content_as_str(path):
    if not os.path.isfile(path):
        r = http.request('GET', path)
        return r.read().decode(r.info().get_content_charset())
    else:
        with open(path, 'r') as f:
            return f.read()


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
    mime = graphene.String(description='Type of the payload described as mime. The text is not mime encoded in this'
                                       'class, but a subclass, such as EmailMessage could change that.',
                           default_value='text/plain')
    text = graphene.String(description='Optional parameter that, if present, eliminates the need to retrieve '
                                       'the text from the URL field.',
                           default_value=None)
    text_features = TextFeatures(description='Optional parameter for classifiers for training with features in '
                                             'addition to the content.',
                                 default_value=None)


class EmailMessage(Text):
    class Meta:
        description = 'Subclass of Text, where text field or document linked is MIME encoded and self describing. ' \
                      'In that case, the "mime" input field is not used.'


class TrainingData(graphene.InputObjectType):
    class Meta:
        description = 'Training data and targets to train a classifier.'
    data = graphene.List(Text, description='List of text document samples or email messages for training.')
    exclusive_targets = graphene.List(graphene.String,
                                      description='A list of exclusive classes that correspond '
                                                  'positionally to the list of documents.',
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
                                    default_value=None)
    vocab_path = graphene.String(description='Optional path to an alternate vocabulary file. This API currently '
                                             'supports the Stanford GloVe format files.',
                                 default_value=DEFAULTS.VOCAB_FILE)
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
                                      default_value=DEFAULTS.EXCLUSIVE)
    overlapping_classes = graphene.List(graphene.String,
                                        description='A list of strings representing the overlapping classes to be '
                                                    'trained or considered in a trained classifier.',
                                        default_value=DEFAULTS.OVERLAPPING)


class ClassifierInfo(graphene.ObjectType):
    class Meta:
        description = 'Return information about an instantiated classifier, including ID and epoch.'

    classifier_id = graphene.String(description='UUID that allows the server to use a cached '
                                                'classifier without reloading. This should be passed in the '
                                                'ClassifierSpec of subsequent calls. Common UUIDs can be used '
                                                'across servers to identify shared classifiers.',
                                    required=True)
    vocab_path = graphene.String(description='Path to the vocabulary used.')
    num_subject_words = graphene.Int(description='Current number of subject words of the classifer.')
    num_body_words = graphene.Int(description='Current number of body words of the classifer.')
    num_features = graphene.Int(description='Number of features, in addition to content, considered.')
    exclusive_classes = graphene.List(graphene.String, description='A list of the exclusive classes handled by this '
                                                                   'classifier.')
    overlapping_classes = graphene.List(graphene.String,
                                        description='A list of strings representing the overlapping classes to be '
                                                    'trained or considered in a trained classifier.')
    epoch = graphene.Int(description='The number of separate epochs of training this classifer has already had.',
                         default_value=0)


class TrainingSpec(graphene.InputObjectType):
    class Meta:
        description = 'Everything needed to identify and train a classifier, including number of epochs ' \
                      'and learning rate.'
    classifier_id = graphene.String(description='Valid ID of an instantiated classifier.')
    train = TrainingData()
    test = TrainingData(default_value=None)
    epochs = graphene.Int(description='Number of epochs to train with the data passed. If the classifier is already '
                                      'trained, this will add epochs to its training', default_value=5)
    learning_rate = graphene.String(description='The learning rate for training.', default_value=0.001)
    holdout_pct = graphene.Float(description='A percentage of the training data to randomly hold back from training '
                                             'and use to measure progress as well as prevent overfitting.',
                                 default_value=0.0)


class ClassifierQuery(graphene.ObjectType):
    class Meta:
        description = 'The main query for classifying text and emails'

    classifiers = graphene.List(ClassifierInfo, description='A list of all previously instantiated classifiers.')

    classifier = graphene.Field(ClassifierInfo, description='Instantiate an existing classifier.',
                                classifier_id=graphene.String(required=True))

    classifications = graphene.List(TextClasses,
                                    description='Classifications of content, using an instantiated classifier.',
                                    classifier_id=graphene.String(required=True),
                                    data=graphene.List(Text),
                                    receiver_address=graphene.String(default_value=None))

    @staticmethod
    def resolve_classifiers(root, info):
        """
        return all known classifiers
        :param root:
        :param info:
        :return:
        """
        return _global_classifier_threads.submit(ClassifierQuery._resolve_classifiers, root, info).result()

    @staticmethod
    def _resolve_classifiers(root, info):
        """
        return all known classifiers
        :param root:
        :param info:
        :return:
        """
        c_files = [f.strip() for f in next(os.walk(DEFAULTS.META_PATH))[2]]

        classifiers = []
        for fn in c_files:
            with open(os.path.join(DEFAULTS.META_PATH, fn), 'r') as f:
                c_dict = json.loads(f.read())

            classifiers.append(
                ClassifierInfo(
                    classifier_id = fn.split('.')[0],
                    vocab_path = c_dict['vocab_path'],
                    exclusive_classes = c_dict['exclusive_classes'],
                    overlapping_classes = c_dict['overlapping_classes'],
                    num_features = c_dict['num_features'],
                    num_subject_words = c_dict['num_subject_words'],
                    num_body_words = c_dict['num_body_words'],
                    epoch = c_dict['epoch']
                )
            )
        return classifiers

    @staticmethod
    def resolve_classifier(root, info, classifier_id):
        """
        load or create a classifier and return information about it
        :param root:
        :param info:
        :param classifier_id:
        :return:
        """
        return _global_classifier_threads.submit(ClassifierQuery._resolve_classifier,
                                                 root, info, classifier_id).result()

    @staticmethod
    def _resolve_classifier(root, info, classifier_id):
        """
        load or create a classifier and return information about it
        :param root:
        :param info:
        :param classifier_id:
        :return:
        """
        try:
            with open(os.path.join(DEFAULTS.META_PATH,
                                   classifier_id + DEFAULTS.META_EXT), 'r') as f:
                c_dict = json.loads(f.read())

        except FileNotFoundError:
            raise FileNotFoundError('Classifier {} not found'.format(classifier_id))

        c_info = _global_classifier_info.get(classifier_id, None)
        if c_info is None:
            classifier = EmailClassifier(os.path.join(DEFAULTS.VOCAB_PATH, c_dict['vocab_path']),
                                         os.path.join(DEFAULTS.MODEL_PATH, classifier_id + DEFAULTS.MODEL_EXT),
                                         exclusive_classes=c_dict['exclusive_classes'],
                                         overlapping_classes=c_dict['overlapping_classes'],
                                         num_analytics_features=c_dict['num_features'],
                                         num_subject_words=c_dict['num_subject_words'],
                                         num_body_words=c_dict['num_body_words'])

            # get ID and store in our local dict as a cache
            _global_classifiers[classifier_id] = classifier

            c_info = ClassifierInfo(classifier_id=classifier_id,
                                    vocab_path=c_dict['vocab_path'],
                                    exclusive_classes=c_dict['exclusive_classes'],
                                    overlapping_classes=c_dict['overlapping_classes'],
                                    num_features=c_dict['num_features'],
                                    num_subject_words=c_dict['num_subject_words'],
                                    num_body_words=c_dict['num_body_words'],
                                    epoch=classifier.neuralnet.epoch_index)

            _global_classifier_info[classifier_id] = c_info

        return c_info

    @staticmethod
    def resolve_classifications(root, info, classifier_id, data, receiver_address):
        """
        classify a dataset and return the answers.
        :param root: reserved
        :param info: unused
        :param classifier_id:
        :param data:
        :param receiver_address:
        :return:
        """
        return _global_classifier_threads.submit(ClassifierQuery._resolve_classifications,
                                                 root, info, classifier_id, data, receiver_address).result()

    @staticmethod
    def _resolve_classifications(root, info, classifier_id, data, receiver_address):
        _ = ClassifierQuery._resolve_classifier(root, info, classifier_id)
        classifier = _global_classifiers.get(classifier_id, None)
        if data is None or len(data) <= 0:
            raise IndexError('Invalid data parameter for classification')

        if isinstance(data[0], EmailMessage):
            class_vectors = classifier.classify(
                [email.message_from_bytes(get_content_as_str(em.url)) if em.text is None else
                 email.message_from_string(em.text) for em in data],
                features=None if data[0].text_features is None else
                [np.array(em.text_features.features) for em in data],
                receiver_address=receiver_address,
                inference=True
            )
        else:
            class_vectors = classifier.classify(
                [get_content_as_str(em.url) if em.text is None else em.text for em in data],
                features=None if data[0].text_features is None
                else [np.array(em.text_features.features) for em in data],
                inference=True
            )

        return [TextClasses(url=c.url, exclusive=ex, overlapping=ol) for c, (ex, ol) in
                zip(data, classifier.numeric_to_text_classes(class_vectors))]


class CreateClassifier(graphene.Mutation):
    class Meta:
        description = 'This mutation is used to create and instantiate a new classifier.'
    class Arguments:
        classifier_spec = ClassifierSpec()

    classifier_info = graphene.Field(ClassifierInfo)

    @staticmethod
    def mutate(root, info, classifier_spec):
        return _global_classifier_threads.submit(CreateClassifier._mutate,
                                                 root, info, classifier_spec).result()

    @staticmethod
    def _mutate(root, info, classifier_spec):
        """
        create a classifier and return information about it
        :param root:
        :param info:
        :param classifier_spec:
        :return:
        """
        if classifier_spec.classifier_id is None:
            classifier_spec.classifier_id = str(uuid.uuid4())

        if os.path.isfile(os.path.join(DEFAULTS.META_PATH,
                                   classifier_spec.classifier_id + DEFAULTS.MODEL_EXT)):
            raise FileExistsError(
                'Classifier {} already exists. Use different id or delete before re-creating.'.format(
                    classifier_spec.classifier_id
                ))

        classifier = EmailClassifier(os.path.join(DEFAULTS.VOCAB_PATH, classifier_spec.vocab_path),
                                     os.path.join(DEFAULTS.MODEL_PATH,
                                                  classifier_spec.classifier_id + DEFAULTS.MODEL_EXT),
                                     exclusive_classes=classifier_spec.exclusive_classes,
                                     overlapping_classes=classifier_spec.overlapping_classes,
                                     num_analytics_features=classifier_spec.num_features,
                                     num_subject_words=classifier_spec.num_subject_words,
                                     num_body_words=classifier_spec.num_body_words)

        # get and ID and store in our local dict as a cache
        _global_classifiers[classifier_spec.classifier_id] = classifier

        CreateClassifier.classifier_info = ClassifierInfo(classifier_id=classifier_spec.classifier_id,
                                                          vocab_path=classifier_spec.vocab_path,
                                                          num_subject_words=classifier_spec.num_subject_words,
                                                          num_body_words=classifier_spec.num_body_words,
                                                          num_features=classifier_spec.num_features,
                                                          exclusive_classes=classifier_spec.exclusive_classes,
                                                          overlapping_classes=classifier_spec.overlapping_classes,
                                                          epoch=classifier.neuralnet.epoch_index)

        with open(os.path.join(DEFAULTS.META_PATH, classifier_spec.classifier_id + DEFAULTS.META_EXT), 'x') as f:
            f.write(json.dumps(CreateClassifier.classifier_info.__dict__))

        _global_classifier_info[classifier_spec.classifier_id] = CreateClassifier.classifier_info

        return CreateClassifier(classifier_info=CreateClassifier.classifier_info)


class TrainClassifier(graphene.Mutation):
    class Meta:
        description = 'This mutation is used to train an already created classifier.'
    class Arguments:
        spec = TrainingSpec()

    classifier_info = graphene.Field(ClassifierInfo)

    @staticmethod
    def mutate(root, info, spec):
        return _global_classifier_threads.submit(TrainClassifier._mutate,
                                                 root, info, spec).result()

    @staticmethod
    def _mutate(root, info, spec):
        classifier_info = ClassifierQuery._resolve_classifier(root, info, spec.classifier_id)

        # if we failed to load, we would throw an exception past here
        classifier = _global_classifiers[spec.classifier_id]
        assert isinstance(classifier, EmailClassifier)

        # load content for training, if necessary
        if isinstance(spec.train.data[0], EmailMessage):
            classifier.train(
                [email.message_from_bytes(get_content_as_str(em.url)) if em.text is None else
                 email.message_from_string(em.text) for em in spec.train.data],
                [spec.train.exclusive_targets, spec.train.overlapping_targets] if classifier.overlapping_classes else
                [spec.train.exclusive_targets],
                features=None if spec.train.data[0].text_features is None else
                [np.array(em.text_features.features) for em in spec.train.data],
                test_content=None if spec.test is None else
                    [email.message_from_bytes(get_content_as_str(em.url)) if em.text is None else
                     email.message_from_string(em.text) for em in spec.test.data],
                test_targets=None if spec.test is None else
                    [spec.test.exclusive_targets, spec.test.overlapping_targets] if classifier.overlapping_classes else
                    [spec.test.exclusive_targets],
                test_features=None if spec.test is None or spec.test.data[0].text_features is None else
                    [np.array(em.text_features.features) for em in spec.test.data],
                receiver_address=spec.train.receiver_address, serialize=1,
                save_path=os.path.join(DEFAULTS.MODEL_PATH, spec.classifier_id + DEFAULTS.MODEL_EXT),
                holdout_pct=spec.holdout_pct, learning_rate=spec.learning_rate, epochs=spec.epochs)
        else:
            classifier.train(
                [get_content_as_str(em.url) if em.text is None else em.text for em in spec.train.data],
                [spec.train.exclusive_targets, spec.train.overlapping_targets] if classifier.overlapping_classes else
                [spec.train.exclusive_targets],
                features=None if spec.train.data[0].text_features is None else
                [np.array(em.text_features.features) for em in spec.train.data],
                test_content=None if spec.test is None else
                    [get_content_as_str(em.url) if em.text is None else em.text for em in spec.test.data],
                test_targets=None if spec.test is None else
                    [spec.test.exclusive_targets, spec.test.overlapping_targets] if classifier.overlapping_classes else
                    [spec.test.exclusive_targets],
                test_features=None if spec.test is None or spec.test.data[0].text_features is None else
                    [np.array(em.text_features.features) for em in spec.test.data],
                serialize=1, save_path=os.path.join(DEFAULTS.MODEL_PATH, spec.classifier_id + DEFAULTS.MODEL_EXT),
                holdout_pct=spec.holdout_pct, learning_rate=spec.learning_rate, epochs=spec.epochs)

        # store current epoch in return and update metadata
        classifier_info.epoch = classifier.neuralnet.epoch_index
        with open(os.path.join(DEFAULTS.META_PATH, spec.classifier_id + DEFAULTS.META_EXT), 'w') as f:
            f.write(json.dumps(classifier_info.__dict__))

        return TrainClassifier(classifier_info=classifier_info)


class DeleteClassifier(graphene.Mutation):
    class Meta:
        description = 'This mutation is used to delete an existing classifier and cleanup its resources.'
    class Arguments:
        classifier_id = graphene.String(description='ID of the classifier to delete.')

    result = graphene.String(description='Return code for delete operation. Success == "OK"')

    @staticmethod
    def mutate(root, info, classifier_id):
        return _global_classifier_threads.submit(DeleteClassifier._mutate,
                                                 root, info, classifier_id).result()

    @staticmethod
    def _mutate(root, info, classifier_id):
        # remove classifier if it is in memory
        try:
            del _global_classifiers[classifier_id]
            del _global_classifier_info[classifier_id]
        except KeyError:
            pass

        try:
            os.remove(os.path.join(DEFAULTS.MODEL_PATH, classifier_id + DEFAULTS.MODEL_EXT))
        except FileNotFoundError:
            pass

        try:
            os.remove(os.path.join(DEFAULTS.TRAIN_PATH, classifier_id + DEFAULTS.TRAIN_EXT))
        except FileNotFoundError:
            pass

        # remove metadata and model from storage, metadata not found can return an error
        try:
            os.remove(os.path.join(DEFAULTS.META_PATH, classifier_id + DEFAULTS.META_EXT))
        except FileNotFoundError:
            raise FileNotFoundError('Classifier {} not found.'.format(classifier_id))

        try:
            os.remove(os.path.join(DEFAULTS.MODEL_PATH, classifier_id + DEFAULTS.MODEL_EXT))
        except FileNotFoundError:
            pass

        return DeleteClassifier(result='OK')


class Mutations(graphene.ObjectType):
    create_classifier = CreateClassifier.Field()
    train_classifier = TrainClassifier.Field()
    delete_classifier = DeleteClassifier.Field()


class ObserveClassifier(graphene.ObjectType):
    class Meta:
        description = 'Provides the ability to subscribe to real-time classifier training data streams.'

    place_holder = graphene.Int()

    def resolve_place_holder(root, info):
        pass


def main():
    schema = graphene.Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier)
    result = schema.execute('query {classifiers { classifierId vocabPath numSubjectWords numBodyWords '
                            'numFeatures exclusiveClasses overlappingClasses epoch }}')
    x = result.data
    print(x)


if __name__ == '__main__':
    main()