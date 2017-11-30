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
import os
import email
from email.utils import getaddresses
import numpy as np
from .neuralnetwork import ClassifierNetwork
from neon.layers import Multicost, GeneralizedCost
from neon.transforms import CrossEntropyMulti, SumSquared
from neon.transforms.cost import MultiMetric, Misclassification, LogLoss
from neon.callbacks.callbacks import Callbacks
from neon.optimizers import Adam
from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import deque
import quopri
import base64
import re
import uuid
from zmlcore.data.dataiterator import BatchIterator

_vocabularies = {}

class EmailClassifier(object):
    def __init__(self, vocab_path, model_path, optimizer=Adam(), overlapping_classes=None, exclusive_classes=None,
                 class_threshold=0.6, num_analytics_features=4, num_subject_words=8, num_body_words=52,
                 network_type='conv_net', name=str(uuid.uuid4())):
        """
        loads the vocabulary and sets up LSTM networks for classification.
        """
        self.name = name
        self.wordvec_dimensions = 0
        self.vocab = self.load_vocabulary(vocab_path)
        self.recurrent = network_type == 'lstm'
        assert self.wordvec_dimensions > 0

        self.num_subject_words = num_subject_words
        self.num_body_words = num_body_words
        self.num_words = num_subject_words + num_body_words
        self.class_threshold = class_threshold

        self.neuralnet = ClassifierNetwork(overlapping_classes=overlapping_classes,
                                           exclusive_classes=exclusive_classes,
                                           optimizer=optimizer, network_type=network_type,
                                           analytics_input=False if num_analytics_features == 0 else True,
                                           num_words=self.num_words, width=self.wordvec_dimensions)

        if not model_path is None:
            try:
                self.neuralnet.load_params(model_path)
            except Exception as e:
                print('{}:{} - cannot load model file {}'.format(type(e), e, model_path))

        if network_type == 'lstm':
            self.zero_tensors = [self.be.zeros((self.wordvec_dimensions, num_subject_words + num_body_words))]
            self.zeros = self.zero_tensors[0][:, 0].get()
            self.zeros = self.zeros.reshape((len(self.zeros)))
        else:
            self.zero_tensors = [self.be.zeros((1, num_subject_words + num_body_words, self.wordvec_dimensions))]
            self.zeros = self.zero_tensors[0][:, 0, :].get()[0]

        # don't add an analytics tensor if we're content only
        if num_analytics_features > 0:
            self.zero_tensors += [self.be.zeros((num_analytics_features, 1))]

        # only add an overlapping classifier if needed
        if overlapping_classes is None:
            self.cost = GeneralizedCost(CrossEntropyMulti())
            self.neuralnet.initialize(self.zero_tensors[0].shape, cost=self.cost)
        else:
            self.cost = Multicost([GeneralizedCost(SumSquared()), GeneralizedCost(CrossEntropyMulti())])
            self.neuralnet.initialize([t.shape for t in self.zero_tensors], cost=self.cost)

    def fit(self, dataset, optimizer, num_epochs, callbacks):
        self.neuralnet.fit(dataset, self.cost, optimizer, num_epochs, callbacks)

    @property
    def be(self):
        """
        return the Nervana backend object
        :return:
        """
        return self.neuralnet.be

    def load_vocabulary(self, vocab_path):
        """
        path for word2vec or glove text embedding vocabularies
        :param vocab_path:
        :return:
        """
        vocab_path = os.path.abspath(vocab_path)
        vocab = _vocabularies.get(vocab_path, None)
        if vocab:
            for wv in vocab.values():
                self.wordvec_dimensions = len(wv)
                break
        else:
            print('loading word vectors from {} ... '.format(vocab_path), end='', flush=True)
            vocab = {}
            with open(vocab_path, 'r') as f:
                tokens = []
                for line in f:
                    if len(line) > 0:
                        tokens = line.split()
                        vocab[tokens[0]] = np.array(tokens[1:], dtype=np.float64)
                self.wordvec_dimensions = len(tokens) - 1
            print('loaded successfully')
            _vocabularies[vocab_path] = vocab
        return vocab

    @property
    def overlapping_classes(self):
        return self.neuralnet.overlapping_classes

    @property
    def exclusive_classes(self):
        return self.neuralnet.exclusive_classes

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
        soup = BeautifulSoup(body, 'html.parser')
        text = ' '.join([s for s in soup.strings if self.tag_visible(s)])
        return RE_SPACES.sub(' ', text)

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
                    result = ' '.join([s.group(0).lower() for s, _ in zip(re.finditer(r"\w+|[^\w\n\s]", text),
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
        text_vectors = [v for v, _ in zip((self.vocab[w] for
                                           w in (s.group(0).lower() for s in re.finditer(r"\w+|[^\w\s]", text))
                                           if not self.vocab.get(w, None) is None),
                                          range(self.num_words))]
        # text_w = [s.group(0).lower() for s, i in zip(re.finditer(r"\w+|[^\w\s]", text), range(self.num_words))]
        # text_v = [self.vocab[w] for w in text_w if not self.vocab.get(w, None) is None]
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
                subject_key: c.get(subject_key, '').lower() if isinstance(c.get(subject_key, ''), str) else '',
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

                subject_v = [v for v, _ in zip((self.vocab[w] for
                                                w in (s.group(0).lower() for s in re.finditer(r"\w+|[^\w\s]", subject))
                                                if not self.vocab.get(w, None) is None),
                                               range(self.num_subject_words))]

                body_v = [v for v, _ in zip((self.vocab[w] for
                                             w in (s.group(0).lower() for s in re.finditer(r"\w+|[^\w\s]", c[text_key]))
                                             if not self.vocab.get(w, None) is None),
                                            range(self.num_body_words))]

                rinputs += subject_v + \
                           [self.zeros for _ in range(np.maximum(int(0), self.num_subject_words - len(subject_v)))] + \
                           body_v + \
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

        dataset = BatchIterator(content, steps=[self.num_words] if features is None else [self.num_words, 1])

        classes = []
        for mb in dataset:
            mb_classes = self.neuralnet.fprop(mb, inference=inference)
            if isinstance(mb_classes, list):
                classes += [[o[:, i].get() for o in mb_classes] for i in range(self.be.bsz)]
            else:
                classes += [mb_classes[:, i].get() for i in range(self.be.bsz)]

        return classes[:len(content)]

    def train(self, content, targets, features=None, receiver_address=None,
              serialize=0, save_path=None, model_file=None, holdout_pct=0.0, learning_rate=0.001, epochs=5):
        """

        :param content: this comes in the form of either emails or plain text to train on
        :param targets: this is either a list of one list of exclusive classes corresponding to each content sample, or
        a list of two lists, that one, and a second that also corresponds to the training samples but with each
        entry a list of overlapping class names
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
        assert isinstance(content, list) and holdout_pct >= 0.0 and holdout_pct < 1.0 and not self.recurrent
        content = self.content_to_nn_representation(content, features=features, receiver_address=receiver_address)
        content[0] = content[0].reshape((len(content[0]), 1, self.num_words, len(content[0][0])))

        # now:
        # classes = list of the two class lists for exclusive and overlapping or just exclusive and None
        # targets = a list containing one list of single class names for exclusive classes or
        #   that + another list of lists of class names for overlapping classes
        ex_targets = [np.array([1 if xt == self.neuralnet.exclusive_classes[i]
                                else 0 for i in range(len(self.neuralnet.exclusive_classes))], dtype=np.float32)
                      for xt in targets[0]]

        # if we also have overlapping targets, add them
        have_ol = not self.neuralnet.overlapping_classes is None
        if have_ol:
            targets = [ex_targets,
                       [np.array([1 if self.neuralnet.overlapping_classes[i] in xt
                                  else 0 for i in range(len(self.neuralnet.overlapping_classes))], dtype=np.float32)
                        for xt in targets[1]]]
        else:
            targets = [ex_targets]

        # create holdout set here if requested
        valid = None

        # if necessary, shuffle and split holdout set
        if holdout_pct > 0.0:
            train_idxs = np.arange(len(targets[0]))
            np.random.shuffle(train_idxs)
            holdout_len = int(len(train_idxs) * holdout_pct)
            if holdout_len > 0:
                holdout_idxs = train_idxs[:holdout_len]
                train_idxs = train_idxs[holdout_len:]

                valid = [np.take(c, holdout_idxs, axis=0) for c in content]
                valid_targets = [np.take(t, holdout_idxs, axis=0) for t in targets]
                valid = BatchIterator(valid,
                                      targets=valid_targets,
                                      steps=[self.num_words] if features is None else [self.num_words, 1])

                content = [np.take(c, train_idxs, axis=0) for c in content]
                targets = [np.take(t, train_idxs, axis=0) for t in targets]

        train = BatchIterator(content,
                              targets=targets,
                              steps=[self.num_words] if features is None else [self.num_words, 1])

        callbacks = Callbacks(self.neuralnet, train_set=train, multicost=have_ol,
                              metric=MultiMetric(Misclassification(), 0) if have_ol else Misclassification(),
                              eval_freq=None if valid is None else 1, eval_set=valid,
                              save_path=save_path, serialize=serialize, model_file=model_file)

        print('Training neural networks on {} samples for {} epochs.'.format(len(targets[0]), epochs))
        self.fit(train, Adam(learning_rate=learning_rate), epochs, callbacks)

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

        if isinstance(classes[0], list) or isinstance(classes[0], tuple):
            # assume exclusive followed by overlapping for now
            assert len(classes[0]) == 2
            return [[self.neuralnet.exclusive_classes[np.argmax(nc[0])],
                     [self.overlapping_classes[i] for i in range(len(self.overlapping_classes))
                      if nc[1][i] > self.class_threshold]]
                    for nc in classes]
        else:
            return [self.neuralnet.exclusive_classes[np.argmax(nc)] for nc in classes]
