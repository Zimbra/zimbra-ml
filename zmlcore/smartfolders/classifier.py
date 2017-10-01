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
from neon.transforms import CrossEntropyMulti, SumSquared
from neon.optimizers import Adam
from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import deque
import quopri
import base64
import re

class EmailClassifier(object):
    def __init__(self, vocab_path, model_path, optimizer=Adam(), overlapping_classes=None, exclusive_classes=None,
                 num_analytics_features=4, num_subject_words=8, num_body_words=52, network_type='conv_net'):
        """
        loads the vocabulary and sets up LSTM networks for classification.
        """
        self.wordvec_dimensions = 0
        self.vocab = self.load_vocabulary(vocab_path)
        self.recurrent = network_type == 'lstm'
        assert self.wordvec_dimensions > 0

        self.num_subject_words = num_subject_words
        self.num_body_words = num_body_words
        self.num_words = num_subject_words + num_body_words

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
            self.neuralnet.initialize(self.zero_tensors[0], cost=self.cost)
        else:
            self.cost = Multicost([GeneralizedCost(SumSquared()), GeneralizedCost(CrossEntropyMulti())])
            self.neuralnet.initialize(self.zero_tensors, cost=self.cost)

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
        return vocab

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)

    def visible_texts(self, body, charset):
        """ get visible text from a document """
        INVISIBLE_ELEMS = ('style', 'script', 'head', 'title')
        RE_SPACES = re.compile(r'\s{3,}')

        soup = BeautifulSoup(body, 'html.parser')
        text = ' '.join([
            s for s in soup.strings
            if s.parent.name not in INVISIBLE_ELEMS
        ])
        # collapse multiple spaces to a space
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
                        text = self.visible_texts(payload, part.get_content_charset())
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

    def emails_to_nn_representation(self, emails, receiver_address=None):
        """
        this converts a list of emails into a list of two vector tuples, each tuple has one element that
        represents the text of the email, and one that represents the additional features, both in the form of
        neural network ready tensors.
        :returns:
        """
        assert isinstance(emails, list) and isinstance(emails[0], email.message.Message)
        from_key = 'from'
        to_key = 'to'
        cc_key = 'cc'
        resent_key = 'resent'
        subject_key = 'subject'
        text_key = 'text'

        emails = [{
            from_key: getaddresses(e.get_all(from_key, [])),
            to_key: getaddresses(e.get_all(to_key, [])),
            cc_key: getaddresses(e.get_all(cc_key, [])),
            resent_key: getaddresses(e.get_all('resent-to', []) + e.get_all('resent-cc', [])),
            subject_key: e.get(subject_key, '').lower() if isinstance(e.get(subject_key, ''), str) else '',
            text_key: ''.join([self.extract_inline_text(p) for p in e.walk()])
                if e.is_multipart() else self.extract_inline_text(e)
        } for e in emails]

        # we index recurrent inputs for steps, so make it a list
        rinputs = []
        linputs = deque()

        for e in emails:
            # until we get contact analytics, we will generate the following contact related features from what we
            # currently do have:
            #   1. is this exclusively to me?
            #   2. is this a reply to an email I sent?
            #   3. am I on the "to" line?
            #   4. is this a forward?
            subject = e[subject_key]

            subject_v = [v for v, _ in zip((self.vocab[w] for
                                            w in (s.group(0).lower() for s in re.finditer(r"\w+|[^\w\s]", subject))
                                            if not self.vocab.get(w, None) is None),
                                           range(self.num_subject_words))]

            body_v = [v for v, _ in zip((self.vocab[w] for
                                         w in (s.group(0).lower() for s in re.finditer(r"\w+|[^\w\s]", e[text_key]))
                                         if not self.vocab.get(w, None) is None),
                                        range(self.num_body_words))]

            rinputs += subject_v + \
                       [self.zeros for _ in range(np.maximum(int(0), self.num_subject_words - len(subject_v)))] + \
                       body_v + \
                       [self.zeros for _ in range(np.maximum(int(0), self.num_body_words - len(body_v)))]

            if not self.neuralnet.overlapping_classes is None:
                if receiver_address:
                    linputs.append([1.0 if (len(e[to_key]) == 1 and receiver_address == e[to_key][0]) else 0.0,
                                    1.0 if ('re:' in subject and receiver_address in to_key) else 0.0,
                                    1.0 if receiver_address in to_key else 0.0,
                                    1.0 if ('fw:' in subject or 'fwd:' in subject) else 0.0])
                else:
                    linputs.append([0.0, 0.0, 0.0, 1.0 if 'fw:' in subject else 0.0])

        return [np.array(rinputs), np.array(linputs)]

    def classify(self, emails, receiver_address=None, inference=False):
        """
        returns classification vectors for all emails in the passed list
        email in the list can either be in neural network representation or as email.message, but all
        must be in the same format. if provided in email.message format, they will be processed into neural
        network representations before classification
        :param emails:
        :return: list of outputs as numpy arrays
        """
        assert isinstance(emails, list)
        if isinstance(emails[0], email.message.Message):
            emails = self.emails_to_nn_representation(emails, receiver_address=receiver_address)

        return [[o.get() for o in x] for x in [[self.neuralnet.fprop(b.nn_input, inference=inference)]
                                               if self.neuralnet.overlapping_classes is None
                                               else self.neuralnet.fprop(b.nn_input, inference=inference)
                                               for b in emails]]
