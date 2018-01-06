"""
created: 9/8/2017, Michael Toutonghi
(c) copyright 2017 Synacor, Inc

Machine learning classifier for email.

Initial implementation leverages primarily subject, address blocks, and body text. It performs the following processing:

1. convert some maximum number of subject and body words to the text to analyze
2. tokenize n words of subject and first o words of text for embed_dim x (n + o) input
3. calculate contact/domain specific features (samedomain, sentto_recently, sentto_semirecently,
    relative_response_to_sender

Create neural networks with:
    1. convolutional or LSTM input for text
    2. linear input for contact/domain specific features
    3. linear blending layer
    4. logistic and softmax classification outputs for overlapping / exclusive classification

This also supports both text and HTML parsing in email, stripping HTML to the visible text only for processing.

"""
import email
from email.utils import getaddresses
import numpy as np
from .neuralnetwork import ClassifierNetwork
from neon.layers import Multicost, GeneralizedCost
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti, SumSquared, MeanSquared, SmoothL1Loss
from neon.transforms.cost import MultiMetric, Misclassification
from neon.callbacks.callbacks import Callbacks
from neon.optimizers import Adam
from neon.backends import gen_backend
from neon.util.argparser import extract_valid_args
from zmlcore.neonfixes.transforms import fix_logistic
from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import deque
import quopri
import base64
import re
import uuid
from zmlcore.data.dataiterator import BatchIterator
from zmlcore.classifier.traincallbacks import MisclassificationTest, LogLossTest
from zmlcore.neonfixes.metrics import AverageLogLoss
from zmlcore.classifier.vocabularies import Vocabularies, clean_text


class Config:
    options = None
    initialized = False

    @staticmethod
    def initialize_neon():
        # for now, we don't trust the mkl backend
        if Config.options.backend == 'mkl':
            print('Resetting mkl backend to cpu')
            Config.options.backend = 'cpu'

        be = gen_backend(**extract_valid_args(Config.options, gen_backend))
        # patch a fix to stabilize the CPU version of Neon's logistic function
        fix_logistic(be)
        Config.initialized = True


class TextClassifier(object):
    def __init__(self, vocab_path, model_path, optimizer=Adam(), overlapping_classes=None, exclusive_classes=None,
                 class_threshold=0.6, num_analytics_features=4, num_subject_words=8, num_body_words=52,
                 network_type='conv_net', preserve_case=False, lookup_size=0, lookup_dim=0,
                 regex=r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|"
                       r":\s?D|:-D|x-?D|X-?D|"
                       r";\)|;-\)|\(-;|\(;|;D|;-D|"
                       r"<3|:\*|;\*|"
                       r":\(|:-\(|\):|\)-:|"
                       r":'\(|:,\(|:\"\(|"
                       r"\(\(\(|\)\)\)|X-\(|\)-X|:\s?@|:-@|@\s?:|@-:|>:\(|\):<|" +
                       '\U0001F620|' + '\U0001F595|' + '\U0001F612|' + '\U0001F608|' + '\U0001F480|' +
                       '\U0001F4A2|' + '\U0001F4A3|' + '\u2620|' + '\uFE0F|' +
                       r"\w+|[^\w\s\n]+)",
                 name=str(uuid.uuid4())):
        """
        :param vocab_path:
        :param model_path:
        :param optimizer:
        :param overlapping_classes:
        :param exclusive_classes:
        :param class_threshold:
        :param num_analytics_features:
        :param num_subject_words:
        :param num_body_words:
        :param network_type:
        :param lookup_size: if zero, we expect to be able to load a vocabulary, otherwise, we will make a lookup
        table that grows as we train
        :param lookup_dim: dimensions of a lookup table if we have one
        :param name:
        """
        assert (overlapping_classes is not None) or (exclusive_classes is not None)

        self.name = name
        self.num_subject_words = num_subject_words
        self.num_body_words = num_body_words
        self.num_words = num_subject_words + num_body_words
        self.class_threshold = class_threshold

        self.lookup_dim = lookup_dim
        self.lookup_size = lookup_size

        self.preserve_case = preserve_case

        self.regex = regex

        self.vocab = Vocabularies.load_vocabulary(vocab_path)
        self.vocab_path = vocab_path

        if self.vocab is None:
            if lookup_size > 0:
                self.wordvec_dimensions = 1
            else:
                self.wordvec_dimensions = 0
        else:
            for wv in self.vocab.values():
                self.wordvec_dimensions = len(wv)
                break

        assert self.wordvec_dimensions > 0

        self.recurrent = network_type == 'lstm'

        self.exclusive_classes = exclusive_classes
        self.overlapping_classes = overlapping_classes

        self.neuralnet_params = {'overlapping_classes':overlapping_classes,
                                 'exclusive_classes':exclusive_classes,
                                 'optimizer':optimizer, 'network_type':network_type,
                                 'analytics_input':False if num_analytics_features == 0 else True,
                                 'num_words':self.num_words, 'width': self.wordvec_dimensions,
                                 'lookup_size':lookup_size, 'lookup_dim': lookup_dim}

        self.model_path = model_path
        self.network_type = network_type
        self.num_features = num_analytics_features

        self.neuralnet = None
        self.optimizer = optimizer
        self.initialize_neural_network()

    def initialize_neural_network(self):
        """
        We put this in to replace the backend with a new one in hopes of clearing the memory used by
        Intel's Neon backend between epochs. It should allow us to replace backends for other reasons, such as
        changing batch size or CPU/GPU as well.
        :return:
        """
        Config.initialize_neon()

        self.neuralnet = ClassifierNetwork(**self.neuralnet_params)

        if not self.model_path is None:
            try:
                self.neuralnet.load_params(self.model_path)
                print('model loaded at epoch {}'.format(self.neuralnet.epoch_index))
            except Exception as e:
                print('{}:{} - cannot load model file {}'.format(type(e), e, self.model_path))

        if self.network_type == 'lstm':
            self.zero_tensors = [self.be.zeros((self.wordvec_dimensions, self.num_words))]
            self.zeros = self.zero_tensors[0][:, 0].get()
            self.zeros = self.zeros.reshape((len(self.zeros)))
        else:
            self.zero_tensors = [self.be.zeros((1, self.num_words, self.wordvec_dimensions))]
            self.zeros = self.zero_tensors[0][:, 0, :].get()[0]

        # use multi-cost metric if we have both exclusive and overlapping classes
        if self.overlapping_classes is not None and self.exclusive_classes is not None:
            self.cost = Multicost([GeneralizedCost(CrossEntropyMulti()),
                                   GeneralizedCost(CrossEntropyBinary())])
        elif self.overlapping_classes is None:
            self.cost = GeneralizedCost(CrossEntropyMulti())
        else:
            self.cost = GeneralizedCost(CrossEntropyBinary())

        # don't add an analytics tensor if we're content only
        if self.num_features > 0:
            self.zero_tensors += [self.be.zeros((self.num_features, 1))]
            self.neuralnet.initialize([t.shape for t in self.zero_tensors], cost=self.cost)
        else:
            self.neuralnet.initialize(self.zero_tensors[0].shape, cost=self.cost)

    def fit(self, dataset, optimizer, num_epochs, callbacks):
        self.neuralnet.fit(dataset, self.cost, optimizer, num_epochs, callbacks)

    @property
    def be(self):
        """
        return the Nervana backend object
        :return:
        """
        return self.neuralnet.be

    def set_case(self, s):
        if self.preserve_case:
            return s
        else:
            return s.lower()

    def tag_visible(self, element):
        if element.parent.name in ('style', 'script', 'head', 'title', 'meta', '[document]'):
            return False
        if isinstance(element, Comment):
            return False
        return True

    def visible_text(self, body):
        """
        return visible text from an HTML document as one python string
        :param body: the HTML document
        :return:
        """
        RE_SPACES = re.compile(r'\s{3,}')
        RE_DUPS = re.compile(r"(\w)\1{3,}")
        soup = BeautifulSoup(body, 'html.parser')
        text = RE_SPACES.sub(' ', ' '.join([s for s in soup.strings if self.tag_visible(s)]))
        return RE_DUPS.sub(r'\1\1\1', text)

    def extract_inline_text(self, part):
        """
        if a part of a message is text, this extracts and cleans the text. if not, it returns an empty string
        both 'text/plain' and 'text/html' are returned as plain text in string form
        :return:
        """
        if 'attachment' in str(part['Content-Disposition']):
            return ''

        ct = part.get_content_type()
        if ct in ['text/plain', 'text/html']:
            te = part['Content-Transfer-Encoding']
            try:
                if te == 'quoted-printable':
                    payload = quopri.decodestring(part.get_payload(decode=True))
                elif te == 'base64':
                    payload = base64.b64decode(part.get_payload())
                else:
                    payload = part.get_payload()

                if ct == 'text/plain':
                    try:
                        result = payload.decode(part.get_content_charset())
                    except:
                        try:
                            result = payload.decode('utf-8')
                        except Exception as e:
                            result = ''
                    return result
                elif ct == 'text/html':
                    try:
                        text = self.visible_text(payload)
                    except Exception as e:
                        return ''
                    result = ' '.join([self.set_case(s.group(0)) for s, _ in zip(re.finditer(self.regex, text),
                                                                          range(self.num_words))])
                    return result
                else:
                    return ''
            except Exception as e:
                print('WARNING: exception {}: {} parsing part'.format(type(e), e))
                return ''
        else:
            return ''

    def text_to_nn_representation(self, text):
        """
        this simply converts text to input for the neural network. it only converts
        the first words up to the total number of subject and body words together.
        this is currently used for testing classification on non-email tests.
        :param text:
        :return:
        """
        text = clean_text(text)
        text_vectors = [v for v, _ in zip((self.vocab[w] for
                                           w in (self.set_case(s.group(0)) for s in re.finditer(self.regex, text))
                                           if not self.vocab.get(w, None) is None),
                                          range(self.num_words))]
        return text_vectors + [self.zeros for _ in range(self.num_words - len(text_vectors))]

    def content_to_nn_representation(self, content, features=None, receiver_address=None):
        """
        this converts a list of emails into a list of two vector tuples, each tuple has one element that
        represents the text of the email, and one that represents the additional features, both in the form of
        neural network ready tensors.
        :returns:
        """
        assert isinstance(content, list) and \
               (isinstance(content[0], email.message.Message) or
                isinstance(content[0], str))
        assert receiver_address is None or features is None
        assert features is None or len(features) == len(content)

        subject_key = 'subject'
        text_key = 'text'

        if isinstance(content[0], email.message.Message):
            from_key = 'from'
            to_key = 'to'
            cc_key = 'cc'
            resent_key = 'resent'

            content = [{
                from_key: getaddresses(c.get_all(from_key, [])),
                to_key: getaddresses(c.get_all(to_key, [])),
                cc_key: getaddresses(c.get_all(cc_key, [])),
                resent_key: getaddresses(c.get_all('resent-to', []) + c.get_all('resent-cc', [])),
                subject_key: self.set_case(c.get(subject_key, '')) if isinstance(c.get(subject_key, ''), str) else '',
                text_key: ''.join([self.extract_inline_text(p) for p in c.walk()])
                    if c.is_multipart() else self.extract_inline_text(c)
            } for c in content]

            rinputs = []
            linputs = deque()

            for c, i in zip(content, range(len(content))):
                # until we get contact analytics, we will generate the following contact related features from what we
                # currently do have:
                #   1. is this exclusively to me?
                #   2. is this a reply to an email I sent?
                #   3. am I on the "to" line?
                #   4. is this a forward?
                subject = c[subject_key]

                subject_v = [v for v, _ in zip((self.vocab[w] for w in
                                                (self.set_case(s.group(0)) for s in re.finditer(self.regex, subject))
                                                if not self.vocab.get(w, None) is None),
                                               range(self.num_subject_words))]

                body_v = [v for v, _ in zip((self.vocab[w] for w in
                                             (self.set_case(s.group(0)) for s in re.finditer(self.regex, c[text_key]))
                                             if not self.vocab.get(w, None) is None),
                                            range(self.num_body_words))]

                rinputs += subject_v + \
                           [self.zeros for _ in range(np.maximum(int(0), self.num_subject_words - len(subject_v)))]\
                           + body_v + \
                           [self.zeros for _ in range(np.maximum(int(0), self.num_body_words - len(body_v)))]

                if not self.neuralnet.analytics_input:
                    if features:
                        linputs.append(np.array(features[i], dtype=np.float32))
                    else:
                        if receiver_address:
                            linputs.append([1.0 if (len(c[to_key]) == 1 and receiver_address == c[to_key][0]) else 0.0,
                                            1.0 if ('re:' in subject and receiver_address in to_key) else 0.0,
                                            1.0 if receiver_address in to_key else 0.0,
                                            1.0 if ('fw:' in subject or 'fwd:' in subject) else 0.0])
                        else:
                            linputs.append([0.0, 0.0, 0.0, 1.0 if 'fw:' in subject else 0.0])

        else:
            rinputs = []
            linputs = deque()

            for c, i in zip(content, range(len(content))):
                rinputs += self.text_to_nn_representation(c)

                if not self.neuralnet.analytics_input:
                    if features:
                        linputs.append(np.array(features[i], dtype=np.float32))

        return [np.array(rinputs), np.array(linputs)] if self.neuralnet.analytics_input else [np.array(rinputs)]

    def classify(self, content, features=None, receiver_address=None, inference=False):
        """
        returns classification vectors for all emails in the passed list
        content in the list can either be as email.message or text string, but all
        must be in the same format.
        :param content:
        :param features:
        :param receiver_address:
        :param inference:
        :return: list of outputs as numpy arrays
        """
        assert isinstance(content, list)
        content = self.content_to_nn_representation(content, features=features, receiver_address=receiver_address)
        if not self.recurrent:
            content[0] = content[0].reshape((len(content[0]), 1, self.num_words, len(content[0][0])))

        dataset = BatchIterator(content, steps=[1] if features is None else [1, 1])

        classes = []
        for mb in dataset:
            mb_classes = self.neuralnet.fprop(mb, inference=inference)
            if isinstance(mb_classes, list):
                classes += [[o[:, i].get() for o in mb_classes] for i in range(self.be.bsz)]
            else:
                classes += [mb_classes[:, i].get() for i in range(self.be.bsz)]

        return classes[:len(content)]

    def gen_training_set(self, content, targets, features=None, test_content=None,
                         test_targets=None, test_features=None, receiver_address=None, holdout_pct=0.0):
        """
        This takes text and text classes, as well as additional features, and an optional holdout percent, and
        converts the input into numerical output that can be used for training.
        :param vocab_path: path name for the vocabulary file, which may need to be generated if it is not in memory
        or file storage
        :param content: list of email.message objects or strings of text
        :param targets: one or two lists, depending on classifier of first exclusive class targets as strings, and
        overlapping class targets as lists of strings, one per content item
        :param features: additional features to input, depending on classifier for each content item
        :param test_content: separate data for testing/validation after each epoch
        :param test_targets: separate targets for testing
        :param test_features: separate features for testing
        :param receiver_address: optional global email address as a feature (legacy, should be deprecated)
        :param holdout_pct: if test_content is None and this is between 0 and 1, a random holdout set will be
        generated from the data for testing and validation.
        :return:
        """
        assert isinstance(content, list) and holdout_pct >= 0.0 and holdout_pct < 1.0 and not self.recurrent

        # if we also have overlapping targets, add them
        have_ol = not self.overlapping_classes is None

        # we may not have a valid vocabulary, which is our first task, if not
        if self.vocab is None:
            # we need to generate a vocabulary from the training data
            if isinstance(content[0], email.message.Message):
                docs = [' '.join([self.set_case(c.get('subject', '')) if isinstance(c.get('subject', ''), str) else '',
                                  ' '.join([self.extract_inline_text(p) for p in c.walk()])
                                  if c.is_multipart() else self.extract_inline_text(c)])
                        for c in content]
            else:
                docs = content

            self.vocab = Vocabularies.gen_vocabulary(self.vocab_path,
                                                     docs,
                                                     self.regex,
                                                     preserve_case=self.preserve_case,
                                                     n_first_words=10000,
                                                     size=self.lookup_size)

        content = self.content_to_nn_representation(content, features=features, receiver_address=receiver_address)

        if not self.recurrent:
            content[0] = content[0].reshape((len(content[0]) // self.num_words, 1, self.num_words, len(content[0][0])))

        if not test_content is None and not test_targets is None:
            holdout_pct = 0.0
            test_content = self.content_to_nn_representation(test_content,
                                                             features=test_features,
                                                             receiver_address=receiver_address)
            test_content[0] = test_content[0].reshape((len(test_content[0]),
                                                       1, self.num_words, len(test_content[0][0])))

            # get the targets with both exclusive and overlapping targets if present
            ol_idx = 0
            ex_targets = None
            ol_targets = None
            if self.exclusive_classes is not None:
                ex_targets = [np.array([1 if xt == self.neuralnet.exclusive_classes[i]
                                        else 0 for i in range(len(self.neuralnet.exclusive_classes))], dtype=np.float32)
                              for xt in test_targets[0]]
                ol_idx += 1
            if self.overlapping_classes is not None:
                ol_targets = [np.array([1 if self.neuralnet.overlapping_classes[i] in xt
                                        else 0 for i in range(len(self.neuralnet.overlapping_classes))],
                                       dtype=np.float32)
                              for xt in test_targets[ol_idx]]
            test_targets = [t for t in [ex_targets, ol_targets] if t is not None]

        ol_idx = 0
        ex_targets = None
        ol_targets = None
        if self.exclusive_classes is not None:
            ex_targets = [np.array([1 if xt == self.neuralnet.exclusive_classes[i]
                                    else 0 for i in range(len(self.neuralnet.exclusive_classes))], dtype=np.float32)
                          for xt in targets[0]]
            ol_idx += 1
        if self.overlapping_classes is not None:
            ol_targets = [np.array([1 if self.neuralnet.overlapping_classes[i] in xt
                                    else 0 for i in range(len(self.neuralnet.overlapping_classes))],
                                   dtype=np.float32)
                          for xt in targets[ol_idx]]
        targets = [t for t in [ex_targets, ol_targets] if t is not None]

        # if necessary, shuffle and split holdout set
        if holdout_pct > 0.0:
            train_idxs = np.arange(len(targets[0]))
            np.random.shuffle(train_idxs)
            holdout_len = int(len(train_idxs) * holdout_pct)
            if holdout_len > 0:
                holdout_idxs = train_idxs[:holdout_len]
                train_idxs = train_idxs[holdout_len:]

                test_content = [np.take(c, holdout_idxs, axis=0) for c in content]
                test_targets = [np.take(t, holdout_idxs, axis=0) for t in targets]

                content = [np.take(c, train_idxs, axis=0) for c in content]
                targets = [np.take(t, train_idxs, axis=0) for t in targets]

        return content, targets, test_content, test_targets


    def train(self, content, targets, test_content=None, test_targets=None, has_features=False,
        serialize=0, save_path=None, learning_rate=0.001, epochs=5):
        """

        :param content: numerical content returned from gen_training_set
        :param targets: numerical targets returned from gen_training_set
        :param test_content: separate test set to be used for evaluation
        :param test_targets: ""
        :param test_features: ""
        :param features: features arte a list of float lists that can also be in string form, but will be converted
        to arrays of floats that must be of the same length as the features specified when creating the classifier.
        :param receiver_address: deprecated in favor of features, but left in for testing
        :param serialize:
        :param save_path:
        :param model_file:
        :param holdout_pct:
        :param learning_rate:
        :param epochs:
        :return:
        """
        multicost = self.exclusive_classes is not None and self.overlapping_classes is not None

        if multicost:
            metric = MultiMetric(Misclassification(), 0)
        elif self.overlapping_classes is None:
            metric = Misclassification()
        else:
            metric = AverageLogLoss()

        print('Training neural networks on {} samples for {} epochs.'.format(len(targets[0]), epochs))

        if not test_content is None and not test_targets is None:
            valid = BatchIterator(test_content,
                                  targets=test_targets,
                                  steps=[1] if has_features else [1, 1])
        else:
            valid = None

        train = BatchIterator(content,
                              targets=targets,
                              steps=[1] if has_features else [1, 1])

        callbacks = Callbacks(self.neuralnet, train_set=train, multicost=multicost, metric=metric,
                              eval_freq=None if valid is None else 1, eval_set=valid)
                              #save_path=save_path, serialize=serialize)
        if serialize:
            callbacks.add_save_best_state_callback(save_path)

        if not valid is None:
            if self.exclusive_classes is not None:
                print('Starting misclassification error = {:.03}%'.format(self.neuralnet.eval(valid, metric)[0] * 100))
                callbacks.add_callback(MisclassificationTest(valid, metric))
            else:
                print('Starting average logloss = {:.04}'.format(self.neuralnet.eval(valid, metric)[0]))
                callbacks.add_callback(LogLossTest(valid, metric))
        if hasattr(self.optimizer, 'learning_rate'):
            self.optimizer.learning_rate = learning_rate
            print('learning rate = {}'.format(learning_rate))
        self.fit(train, self.optimizer, epochs, callbacks)

    def numeric_to_text_classes(self, classes):
        """
        takes a list of numpy arrays or a list of lists of numpy arrays that correspond to the classes
        of our neural network and returns the text classifications to which they correspond as either a list of
        words, or a list of lists, each including one word for the exclusive class and a list of words for the
        overlapping classes.
        :param classes:
        :return:
        """
        assert isinstance(classes, list)

        if self.exclusive_classes is None:
            return [self.overlapping_classes[i] for i in range(len(self.overlapping_classes))
                      for nc in classes if nc[0][i] > self.class_threshold]
        elif isinstance(classes[0], list) or isinstance(classes[0], tuple):
            # assume exclusive followed by overlapping for now
            assert len(classes[0]) == 2
            return [[self.neuralnet.exclusive_classes[np.argmax(nc[0])],
                     [self.overlapping_classes[i] for i in range(len(self.overlapping_classes))
                      if nc[1][i] > self.class_threshold]]
                    for nc in classes]
        else:
            return [self.neuralnet.exclusive_classes[np.argmax(nc)] for nc in classes]
