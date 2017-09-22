"""
created: 9/8/2017
(c) copyright 2017 Synacor, Inc

Machine learning classifier for email.

Initial implementation leverages primarily subject, address blocks, and body text. It performs the following processing:

1. canonicalize subject and hash
2. tokenize 8 words of subject and first 22 words of text for embed_dim x 30 LSTM input
3. calculate contact/domain specific features (samedomain, sentto_recently, sentto_semirecently,
    relative_response_to_sender

create neural networks with:
    1. LSTM input for text
    2. linear input for contact/domain specific features
    3. linear blending layer
    4. softmax classification outputs

"""

import email
from email.utils import getaddresses
import numpy as np
from .neuralnetwork import ClassifierNetwork
from neon.layers import Multicost, GeneralizedCost
from neon.transforms import CrossEntropyMulti, SumSquared
from neon.optimizers import Adam
from bs4 import BeautifulSoup

class EmailClassifier(object):
    def __init__(self, vocab_path, model_path, optimizer=Adam(), overlapping_classes=None, exclusive_classes=None,
                 num_analytics_features=4, num_subject_words=8, num_body_words=22):
        """
        loads the vocabulary and sets up LSTM networks for classification.
        """
        self.neuralnet = ClassifierNetwork(overlapping_classes=overlapping_classes,
                                           exclusive_classes=exclusive_classes,
                                           optimizer=optimizer)
        self.wordvec_dimensions = 0
        self.vocab = self.load_vocabulary(vocab_path)
        assert self.wordvec_dimensions > 0

        if not model_path is None:
            try:
                self.neuralnet.load_params(model_path)
            except Exception as e:
                print('{}:{} - cannot load model file {}'.format(type(e), e, model_path))

        self.zero_tensors = [self.be.zeros((self.wordvec_dimensions, num_subject_words + num_body_words)),
                             self.be.zeros((num_analytics_features, 1))]

        self.num_subject_words = num_subject_words
        self.num_body_words = num_body_words

        self.cost = Multicost([GeneralizedCost(SumSquared()), GeneralizedCost(CrossEntropyMulti())])

        self.neuralnet.initialize(self.zero_tensors, cost=self.cost)

    def fit(self, dataset, optimizer, num_epochs, callbacks):
        self.neuralnet.fit(dataset, self.cost, optimizer, num_epochs, callbacks)

    @property
    def be(self):
        return self.neuralnet.be

    def load_vocabulary(self, vocab_path):
        """
        path for word2vec or glove text embedding vocabularies
        :param vocab_path:
        :return:
        """
        vocab = {}
        with open(vocab_path, 'r') as f:
            tokens = []
            for line in f:
                if len(line) > 0:
                    tokens = line.split()
                    vocab[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
            self.wordvec_dimensions = len(tokens) - 1
        return vocab

    def extract_inline_text(self, part):
        """
        if a part of a message is text, this extracts and cleans the text. if not, it returns an empty string
        both 'text/plain' and 'text/html' are returned as plain text in string form
        :return:
        """
        if 'attachment' in str(part.get('Content-Disposition')):
            return ''

        ct = part.get_content_type()
        if ct == 'text/plain':
            try:
                result = part.get_payload(decode=True).decode(part.get_content_charset())
            except:
                try:
                    result = part.get_payload(decode=True).decode('utf-8')
                except:
                    result = ''
            return result
        elif ct == 'text/html':
            try:
                text = BeautifulSoup(part.get_payload(), 'html.parser').get_text()
            except:
                return ''
            lines=(line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split(' '))
            return ''.join(chunk for chunk in chunks if chunk)
        else:
            return ''

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

        nn_inputs = []

        for e in emails:
            # until we get contact analytics, we will generate the following contact related features from what we
            # currently do have:
            #   1. is this exclusively to me?
            #   2. is this a reply to an email I sent?
            #   3. am I on the "to" line?
            #   4. is this a forward?
            subject = e[subject_key]
            subject_w = [s.lower() for s in
                         subject.split(maxsplit=self.num_subject_words)[:self.num_subject_words]]
            body_w = [s.lower() for s in
                         # e[text_key]['body'].split(maxsplit=self.num_body_words)[:self.num_body_words]]
                         e[text_key].split(maxsplit=self.num_body_words)[:self.num_body_words]]
            zeros = self.zero_tensors[0][:,0].get().transpose()[0]
            recurrent_input = [self.vocab.get(w, zeros) for w in subject_w] + \
                              [zeros for _ in range(self.num_subject_words - len(subject_w))] + \
                              [self.vocab.get(w, zeros) for w in body_w] + \
                              [zeros for _ in range(self.num_body_words - len(body_w))]

            if receiver_address:
                linear_input = [1.0 if (len(e[to_key]) == 1 and receiver_address == e[to_key][0]) else 0.0,
                                1.0 if ('re:' in subject and receiver_address in to_key) else 0.0,
                                1.0 if receiver_address in to_key else 0.0,
                                1.0 if ('fw:' in subject or 'fwd:' in subject) else 0.0]
            else:
                linear_input = [0.0, 0.0, 0.0, 1.0 if 'fw:' in subject else 0.0]

            nn_inputs.append([self.be.array(recurrent_input).transpose(), self.be.array(linear_input)])
        return nn_inputs

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

        assert (isinstance(emails[0], EmailClassifier.NeuralEmailRepresentation))

        return [[o.get(), x.get()] for o, x in [self.neuralnet.fprop(b.nn_input, inference=inference) for b in emails]]
