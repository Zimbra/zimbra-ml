"""
created: 11/1/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

GraphQL schema for the Zimbra email and text classification micro-service.

The general idea is that operations revolve around classifiers. Classifiers take text or email content and floating
point features, as well as labels for training, and either train using a dataset to create a model, or classify.

Classifiers can be created, trained, used for classification and deleted.
Training sets can be created through a training operation with persist, used for training, and deleted.
Models can be created by training and deleted.

"""
import os
import graphene
import urllib3
import numpy as np
import pickle
from zmlcore.smartfolders.classifier import TextClassifier
import uuid
import email
import json
from concurrent.futures import ThreadPoolExecutor
import pytz
import datetime


_version = '0.0.1'

class DEFAULTS:
    MODEL_PATH = 'data/models/'
    MODEL_EXT = '.model'
    VOCAB_PATH = 'data/vocabularies/'
    VOCAB_EXT = '.vocab' # these are for our auto-vocabularies
    VOCAB_FILE = 'glove.6B.100d.txt'
    LOOKUP_SIZE = 0
    LOOKUP_DIM = 0
    META_PATH = 'data/meta/'
    TRAIN_PATH = 'data/train/'
    META_EXT = '.zml'
    TRAIN_EXT = '.train'
    OVERLAPPING = None
    EXCLUSIVE = None
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


def naive_local_to_naive_utc(dt, local_zone):
    if isinstance(local_zone, str):
        local_zone = pytz.timezone(local_zone)
    return local_zone.localize(dt).astimezone(pytz.utc).replace(tzinfo=None)


def datetime_as_datastring(dt):
    assert isinstance(dt, datetime.datetime)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def save_classifier_info(classifier_info):
    d = classifier_info.__dict__.copy()
    d['training_set'] = None if classifier_info.training_set is None else classifier_info.training_set.__dict__
    with open(os.path.join(DEFAULTS.META_PATH, classifier_info.classifier_id + DEFAULTS.META_EXT), 'w') as f:
        f.write(json.dumps(d))


def load_classifier_info(classifier_id):
    try:
        with open(os.path.join(DEFAULTS.META_PATH,
                               classifier_id + DEFAULTS.META_EXT), 'r') as f:
            c_dict = json.loads(f.read())
            c_info = ClassifierInfo(**{k: v for k, v in c_dict.items() if k != 'training_set'},
                                    training_set=None if c_dict.get('training_set', None) is None else
                                    TrainingSetInfo(**c_dict['training_set']))
            return c_info

    except FileNotFoundError:
        raise FileNotFoundError('Classifier {} not found'.format(classifier_id))


class ClassifierOutput(graphene.ObjectType):
    class Meta:
        description = 'Lists of lists of raw numeric data that is the floating point output of the classifier ' \
                      'before it has been sorted, maxxed, and mapped to class names.'
    name = graphene.String(required=True)
    output = graphene.List(graphene.List(graphene.Float),
                           description='List of 1 or more lists of floats for each output type.')


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
    vocab_path = graphene.String(description='Optional name of a vocabulary file. This API currently '
                                             'supports the Stanford GloVe format files. Specifying a lookup table '
                                             'size with lookup_size causes this parameter to be ignored.',
                                 default_value=DEFAULTS.VOCAB_FILE)
    lookup_size = graphene.Int(description='Alternative to vocab_path, used for auto lookup table creation from '
                                           'training data.',
                                 default_value=DEFAULTS.LOOKUP_SIZE)
    lookup_dim = graphene.Int(description='Dimensions of embeddings in the automatic lookup table.',
                                 default_value=DEFAULTS.LOOKUP_DIM)
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


class TrainingSpec(graphene.InputObjectType):
    class Meta:
        description = 'Everything needed to identify and train a classifier, including number of epochs ' \
                      'and learning rate.'
    classifier_id = graphene.String(description='Valid ID of an instantiated classifier.')
    persist = graphene.Boolean(description='If true, this training set and any holdout set split randomly from it '
                                           'will be persisted across multiple training sessions. This requires '
                                           'server storage, but can save network traffic and allow training progress '
                                           'and testing on one common holdout set over time.', default_value=False)
    train = TrainingData(description='If this is None, the classifier must have a valid, persisted training set.',
                         default_value=None)
    test = TrainingData(default_value=None)
    epochs = graphene.Int(description='Number of epochs to train with the data passed. If the classifier is already '
                                      'trained, this will add epochs to its training', default_value=5)
    learning_rate = graphene.Float(description='The learning rate for training.', default_value=0.001)
    holdout_pct = graphene.Float(description='A percentage of the training data to randomly hold back from training '
                                             'and use to measure progress as well as prevent overfitting.',
                                 default_value=0.0)


class TrainingSetInfo(graphene.ObjectType):
    class Meta:
        description = 'Information about a persisted training set that enables one common holdout set to be' \
                      'used repeatedly across separate training sessions'
    date = graphene.String(description='The date and time in UTC format that this training set was persisted.')
    num_train = graphene.Int(description='The number of training samples in this training set', default_value=0)
    num_test = graphene.Int(description='The number of test samples in the holdout/test set for validation',
                            default_value=0)


class SubscriptionMessage(graphene.ObjectType):
    class Meta:
        description = 'Base type for all subscription messages.'
    date = graphene.String(description='The date and time in UTC format that this message was created.')


class EpochTrainingResults(SubscriptionMessage):
    class Meta:
        description = 'Progress results per epoch, validation loss and misclassification rate.'
    batches_complete = graphene.Int(description='Number of batches completed this epoch.')
    loss = graphene.List(graphene.Float, description='Cross entropy validation loss for exclusive classes and MSE '
                                                     'loss for overlapping classes. This returns a list of one or '
                                                     'more values, with one value per output class type.')
    misclassification = graphene.List(graphene.Float, description='Validation misclassification rate in percent. This '
                                                                  'returns a list of one or more values, with one '
                                                                  'value per output class type.')


class BatchTrainingResults(graphene.ObjectType):
    class Meta:
        description = 'Training loss results per batch.'
    loss = graphene.List(graphene.Float, description='Cross entropy training loss for exclusive classes and MSE '
                                                     'loss for overlapping classes. This returns a list of one or '
                                                     'more values, with one value per output class type.')
    misclassification = graphene.List(graphene.Float, description='Training misclassification rate in percent. This '
                                                                  'returns a list of one or more values, with one '
                                                                  'value per output class type.')


class ClassifierInfo(graphene.ObjectType):
    class Meta:
        description = 'Return information about an instantiated classifier, including ID and epoch.'

    classifier_id = graphene.String(description='UUID that allows the server to use a cached '
                                                'classifier without reloading. This should be passed in the '
                                                'ClassifierSpec of subsequent calls. Common UUIDs can be used '
                                                'across servers to identify shared classifiers.',
                                    required=True)
    vocab_path = graphene.String(description='Path to the vocabulary used.')
    lookup_size = graphene.Int(description='Alternative to vocab_path, used for auto lookup table creation from '
                                           'training data.')
    lookup_dim = graphene.Int(description='Dimensions of embeddings in the automatic lookup table.')
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
    training_set = graphene.Field(TrainingSetInfo, default_value=None)


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
            classifiers.append(load_classifier_info(fn.split('.')[0]))

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
        load n existing classifier and return information about it
        :param root:
        :param info:
        :param classifier_id:
        :return:
        """
        c_dict = load_classifier_info(classifier_id).__dict__

        # now check to see if it is instantiated as well
        c_info = _global_classifier_info.get(classifier_id, None)
        if c_info is None:
            classifier = TextClassifier(os.path.join(DEFAULTS.VOCAB_PATH, c_dict['vocab_path']),
                                        os.path.join(DEFAULTS.MODEL_PATH, classifier_id + DEFAULTS.MODEL_EXT),
                                        exclusive_classes=c_dict['exclusive_classes'],
                                        overlapping_classes=c_dict['overlapping_classes'],
                                        num_analytics_features=c_dict['num_features'],
                                        num_subject_words=c_dict['num_subject_words'],
                                        num_body_words=c_dict['num_body_words'],
                                        lookup_size=c_dict['lookup_size'], lookup_dim=c_dict['lookup_dim'])

            # get ID and store in our local dict as a cache
            _global_classifiers[classifier_id] = classifier

            c_info = ClassifierInfo(classifier_id=classifier_id,
                                    vocab_path=c_dict['vocab_path'],
                                    exclusive_classes=c_dict['exclusive_classes'],
                                    overlapping_classes=c_dict['overlapping_classes'],
                                    num_features=c_dict['num_features'],
                                    num_subject_words=c_dict['num_subject_words'],
                                    num_body_words=c_dict['num_body_words'],
                                    epoch=classifier.neuralnet.epoch_index,
                                    training_set=c_dict['training_set'],
                                    lookup_size=c_dict['lookup_size'],
                                    lookup_dim=c_dict['lookup_dim'])

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

        if classifier_spec.lookup_size > 0:
            # if we are to use a lookup table, we generate an internal, per classifier vocabulary path name
            classifier_spec.vocab_path = \
                classifier_spec['vocab_path'] = classifier_spec.classifier_id + DEFAULTS.VOCAB_EXT

        classifier = TextClassifier(os.path.join(DEFAULTS.VOCAB_PATH, classifier_spec.vocab_path),
                                    os.path.join(DEFAULTS.MODEL_PATH,
                                                  classifier_spec.classifier_id + DEFAULTS.MODEL_EXT),
                                    exclusive_classes=classifier_spec.exclusive_classes,
                                    overlapping_classes=classifier_spec.overlapping_classes,
                                    num_analytics_features=classifier_spec.num_features,
                                    num_subject_words=classifier_spec.num_subject_words,
                                    num_body_words=classifier_spec.num_body_words,
                                    lookup_size=classifier_spec.lookup_size,
                                    lookup_dim=classifier_spec.lookup_dim)

        # get and ID and store in our local dict as a cache
        _global_classifiers[classifier_spec.classifier_id] = classifier

        CreateClassifier.classifier_info = ClassifierInfo(classifier_id=classifier_spec.classifier_id,
                                                          vocab_path=classifier_spec.vocab_path,
                                                          num_subject_words=classifier_spec.num_subject_words,
                                                          num_body_words=classifier_spec.num_body_words,
                                                          num_features=classifier_spec.num_features,
                                                          exclusive_classes=classifier_spec.exclusive_classes,
                                                          overlapping_classes=classifier_spec.overlapping_classes,
                                                          epoch=classifier.neuralnet.epoch_index,
                                                          lookup_size=classifier_spec.lookup_size,
                                                          lookup_dim=classifier_spec.lookup_dim)

        save_classifier_info(CreateClassifier.classifier_info)
        _global_classifier_info[classifier_spec.classifier_id] = CreateClassifier.classifier_info
        return CreateClassifier(classifier_info=CreateClassifier.classifier_info)


class TrainClassifier(graphene.Mutation):
    class Meta:
        description = 'This mutation is used to train an already created classifier.'

    class Arguments:
        training_spec = TrainingSpec(description='A specification of this training session')

    classifier_info = graphene.Field(ClassifierInfo)

    @staticmethod
    def mutate(root, info, training_spec):
        return _global_classifier_threads.submit(TrainClassifier._mutate,
                                                 root, info, training_spec).result()

    @staticmethod
    def _mutate(root, info, training_spec):
        classifier_info = ClassifierQuery._resolve_classifier(root, info, training_spec.classifier_id)

        # if we failed to load, we would throw an exception past here
        classifier = _global_classifiers[training_spec.classifier_id]
        assert isinstance(classifier, TextClassifier)

        if training_spec.train is None:
            # either we have a persisted training set, or we throw an error
            if classifier_info.training_set is None:
                raise ValueError('train data is null and there is no persisted training set.')

            with open(os.path.join(DEFAULTS.TRAIN_PATH, training_spec.classifier_id + DEFAULTS.TRAIN_EXT), 'rb') as f:
                training_data = pickle.load(f)

            if training_data['version'] != _version:
                raise ValueError('persisted training data is incorrect format. please delete and recreate classifier.')

            (train_x, train_y) = training_data['train']
            (test_x, test_y) = training_data['test']
        else:
            # load content for training, if necessary
            if isinstance(training_spec.train.data[0], EmailMessage):
                train_x, train_y, test_x, test_y = classifier.gen_training_set(
                    [email.message_from_bytes(get_content_as_str(em.url)) if em.text is None else
                     email.message_from_string(em.text) for em in training_spec.train.data],
                    [t for t in [training_spec.train.exclusive_targets, training_spec.train.overlapping_targets] if
                     t is not None],
                    features=None if training_spec.train.data[0].text_features is None else
                    [np.array(em.text_features.features) for em in training_spec.train.data],
                    test_content=None if training_spec.test is None else
                        [email.message_from_bytes(get_content_as_str(em.url)) if em.text is None else
                         email.message_from_string(em.text) for em in training_spec.test.data],
                    test_targets=None if training_spec.test is None else
                    [t for t in [training_spec.test.exclusive_targets, training_spec.test.overlapping_targets] if
                     t is not None],
                    test_features=None if training_spec.test is None or training_spec.test.data[0].text_features is None else
                        [np.array(em.text_features.features) for em in training_spec.test.data],
                    receiver_address=training_spec.train.receiver_address, holdout_pct=training_spec.holdout_pct)
            else:
                train_x, train_y, test_x, test_y = classifier.gen_training_set(
                    [get_content_as_str(em.url) if em.text is None else em.text for em in training_spec.train.data],
                    [t for t in [training_spec.train.exclusive_targets, training_spec.train.overlapping_targets] if
                     t is not None],
                    features=None if training_spec.train.data[0].text_features is None else
                    [np.array(em.text_features.features) for em in training_spec.train.data],
                    test_content=None if training_spec.test is None else
                        [get_content_as_str(em.url) if em.text is None else em.text for em in training_spec.test.data],
                    test_targets=None if training_spec.test is None else
                        [t for t in [training_spec.test.exclusive_targets, training_spec.test.overlapping_targets] if t is not None],
                    test_features=None if training_spec.test is None or training_spec.test.data[0].text_features is None else
                        [np.array(em.text_features.features) for em in training_spec.test.data],
                    holdout_pct=training_spec.holdout_pct)

                if training_spec.persist:
                    classifier_info.training_set = \
                        TrainingSetInfo(date=datetime_as_datastring(datetime.datetime.utcnow()),
                                        num_train=len(train_y[0]),
                                        num_test=0 if test_y is None else len(test_y[0]))

                    save_classifier_info(classifier_info)

                    with open(os.path.join(DEFAULTS.TRAIN_PATH,
                                           training_spec.classifier_id + DEFAULTS.TRAIN_EXT), 'wb') as f:
                        try:
                            pickle.dump({'version': _version,
                                         'train': (train_x, train_y),
                                         'test': (test_x, test_y)}, f)
                        except Exception as e:
                            print(e)

        classifier.train(train_x, train_y, test_content=test_x, test_targets=test_y, serialize=1,
                         save_path=os.path.join(DEFAULTS.MODEL_PATH, training_spec.classifier_id + DEFAULTS.MODEL_EXT),
                         learning_rate=training_spec.learning_rate, epochs=training_spec.epochs)

        # store current epoch in return and update metadata
        classifier_info.epoch = classifier.neuralnet.epoch_index
        save_classifier_info(classifier_info)

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
            # remove any custom vocabulary, if there is one
            os.remove(os.path.join(DEFAULTS.VOCAB_PATH, classifier_id + DEFAULTS.VOCAB_EXT))
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


class DeleteTrainSet(graphene.Mutation):
    class Meta:
        description = 'This deletes the training set for an existing classifier ' \
                      'to reduce its storage if the training set will not be used again. This leaves any ' \
                      'trained models and classifier itself intact.'
    class Arguments:
        classifier_id = graphene.String(description='ID of the classifier from which to delete the training set.')

    result = graphene.String(description='Return code for delete operation. Success == "OK"')

    @staticmethod
    def mutate(root, info, classifier_id):
        return _global_classifier_threads.submit(DeleteTrainSet._mutate,
                                                 root, info, classifier_id).result()

    @staticmethod
    def _mutate(root, info, classifier_id):
        try:
            # get classifier info to remove the reference
            classifier_info = ClassifierQuery._resolve_classifier(root, info, classifier_id)
            classifier_info.training_set = None
            save_classifier_info(classifier_info)
        except KeyError:
            pass

        try:
            os.remove(os.path.join(DEFAULTS.TRAIN_PATH, classifier_id + DEFAULTS.TRAIN_EXT))
        except FileNotFoundError:
            pass

        return DeleteTrainSet(result='OK')


class DeleteModels(graphene.Mutation):
    class Meta:
        description = 'This deletes any trained models for an existing classifier, but leaves the classifier' \
                      'and any existing training set in place.'
    class Arguments:
        classifier_id = graphene.String(description='ID of the classifier from which to delete the training set.')

    result = graphene.String(description='Return code for delete operation. Success == "OK"')

    @staticmethod
    def mutate(root, info, classifier_id):
        return _global_classifier_threads.submit(DeleteModels._mutate,
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

        return DeleteModels(result='OK')


class Mutations(graphene.ObjectType):
    create_classifier = CreateClassifier.Field()
    train_classifier = TrainClassifier.Field()
    delete_classifier = DeleteClassifier.Field()
    delete_train_set = DeleteTrainSet.Field()
    delete_models = DeleteModels.Field()


class ObserveClassifier(graphene.ObjectType):
    class Meta:
        description = 'Provides the ability to subscribe to real-time classifier training data streams.'

    place_holder = graphene.Int()

    def resolve_place_holder(root, info):
        raise NotImplementedError()


def main():
    schema = graphene.Schema(query=ClassifierQuery, mutation=Mutations, subscription=ObserveClassifier)
    result = schema.execute('query {classifiers { classifierId vocabPath numSubjectWords numBodyWords '
                            'numFeatures exclusiveClasses overlappingClasses epoch }}')
    x = result.data
    print(x)


if __name__ == '__main__':
    main()