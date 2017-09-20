<div>

<span class="c1"></span>

</div>

<span class="c8">Automatic Classification of Messages</span>

# <span class="c2">Overview</span>

<span>As part of the Next Gen Zimbra server, we will implement an automatic message classification system, initially supporting email, which will tag incoming messages with classification metadata based on</span> <span class="c7">[insights published on automatic classification](https://www.google.com/url?q=https://arxiv.org/pdf/1606.09296.pdf&sa=D&ust=1505953497834000&usg=AFQjCNG0klLNefl4csM7fBGvJLfWWNB4bw)</span><span class="c1"> by Yahoo Research.</span>

<span class="c1"></span>

<span class="c1">The core insight from recent research such as Yahoo’s that we will leverage is that a small number of automatic categories can provide significant value to users and maps to the most common way people who take the time already organize their folders. While Yahoo used latent dirichlet allocation (LDA), topic modeling, opt-in user interaction, and a great deal of analytics, we will take a simpler approach that leverages the value of their conclusions, by which it justifies its relatively modest goals.</span>

<span class="c1"></span>

<span class="c1">As a starting point, we will create a general classification system, but focus our efforts and design on the requirement that the system do an acceptable or better job classifying messages, at least in the general categories considered most useful, based on research. We will also make some decisions about how we interpret those categories, which we believe will improve on Yahoo’s approach as well as guide us in defining the data and features needed for our classification tasks.</span>

<span class="c1"></span>

<span class="c1">The broad categories yahoo identified as useful and important in Yahoo’s research were: “human” generated, “shopping”, “financial”, “travel”, “social”, and “career”. While separating out “human” as a category simplified classification using bag of words methods or cosine similarity and allowed for some shortcuts (all “re:” and “fw:” mails were considered human), it would certainly help if an email system could separate human generated messages into those that are more likely to be considered important as well as allow human generated messages to be included in the category “career” or “financial”, as well as potentially any of the others.</span>

# <span class="c2">Zimbra Message Classifier</span>

<span class="c1">In order to improve upon today’s best approaches, make these categories fit the broadest number of users, regardless of their profession or lifestyle, and attempt to provide more organization of all e-mail streams, not just computer generated, we will define the following categories:</span>

<span class="c1"></span>

*   <span class="c1">“Important” - all mails that the system believes the user is likely to consider higher priority or more important than the others. While messages categorized into important will usually be human generated, we may also find that certain types of computer generated mail, such as bank notifications, utilities, bills, permits, or other types of mail is generally considered important/high priority for users.</span>
*   <span class="c1">“Work” - this is intended to serve as a proxy for “career”, since while not everyone will identify with having a career specifically, almost all people, whether entrepreneurs, house husbands/wives, career minded professionals, or CEOs likely have work.</span>
*   <span class="c1">“Shopping” - the promotional spam that you don’t consider spam</span>
*   <span class="c1">“Financial” - this likely includes mail from financial institutions/banks, etc., but may include human generated email from a stock broker, etc.</span>
*   <span class="c1">“Social” - notifications and messages from social networking services</span>

<span class="c1"></span>

<span class="c1">Our approach to classification will be implemented as a postfix filter, leveraging the milter API, which is the same API supported by numerous mail filters, virus, and spam scanners, including AMaVis. We will start with access to the raw message, and leverage the following data in making our classification decision:</span>

<span class="c1"></span>

*   <span class="c1">all address headers, complete email/friendly addresses, to and cc blocks</span>
*   <span class="c1">subject and hash of canonical subject</span>
*   <span class="c1">body text</span>

# <span class="c2">Analytics for Scalable Personalization</span>

<span class="c1">To enable the use of a machine learning model that is common to all users, yet makes decisions based on individual users’ relationships to senders, text/subject and other addresses in the mail address block, certain features used for classification of messages will be based on analytics, which will be processed and gathered regularly for each account. The following are analytics-based features that we expect to use analytics to classify which emails should be categorized as “important”.</span>

<span class="c1"></span>

<span class="c1">Analytics-driven Features</span>

1.  <span class="c1">How many times has user sent e-mail to sender?</span>
2.  <span class="c1">How many times has sender sent e-mail to user?</span>
3.  <span class="c1">1 & 2 for past day, week, month, forever</span>
4.  <span class="c1">Abbreviate #3 for first 10 addresses in to/cc</span>
5.  <span class="c1">Percent email opened from sender, avg time to open email from sender vs. global avg</span>
6.  <span class="c1">Subject canonical hash matches sent mail at any time from user to sender (this may be difficult or require contact analytics of canonical subject hashes to user)</span>

<span class="c1"></span>

<span class="c1">In addition, we will consider the following features, which will be calculated/cached on the fly:</span>

*   <span class="c1">User is on “to”, “cc”, “none/bcc”</span>
*   <span class="c1">Number of other users on “to” on “cc”</span>
*   <span class="c1">Subject canonical hash matches recently sent mail</span>

<span class="c1"></span>

<span class="c1">We will leverage the message content for classification with a simplifying assumption that all of the text required to properly classify the overwhelming majority of e-mail will be contained in the first “n” words of an e-mail’s concatenated subject and body text, where “n” is a small number, such as 20 to 30\. The rationale for this is that anyone sending e-mail that they expect the receiver to read, except for cases where the identity of the sender overwhelmingly takes precedence over the subject, MUST identify their topic very early in the message. Based on the realities of the email volumes, value of time and human attention as applied to email, we will make the assertion that the previous statement is true and expect to use only the first “small n” number of words for classification. The exact value of n will be determined empirically. This assumption enables us to use potentially more effective classification techniques, such as recurrent neural network inference, while limited the compute cost through limiting the amount of data processed. For example, a small recurrent neural network running inference on 30 words, each being a 100 dimensional word embedding (described below) may take tens of milliseconds on a typical, unaccelerated server, and could be vertically (with GPU acceleration) and horizontally scaled to any degree. Running large, variable sequences of words through such a network could quickly become prohibitive, but it becomes an option if our assumption of the first 30 words of content is correct.</span>

<span class="c1"></span>

<span class="c1">Since we expect to attempt classification on a broad range of emails, including those generated by humans, we must be able to understand an incredibly large vocabulary, while remaining localizable to multiple languages. We will use LSTM neural networks rather than rely on TF-IDF or bag of words models for classification, and to deal with the inherently large vocabulary space while still enabling generalized learning of our classifiers, we will employ word embeddings, which provide numeric representations of words that correlate strongly to the meaning of those words and their relationship to others. For example, it is widely cited in word embedding tutorials that in typical embeddings, if you take the vector representation of the word for “king”, subtract the vector of the word for “man”, and add the vector of the word for “woman”, the resulting vector will be extremely close to the vector for “queen”. While we will endeavor to train adequately across the types of mail we expect to classify, the use of word embeddings should improve generalization across terms or combinations of terms that we may not have previously encountered.</span>

<span class="c1"></span>

<span class="c1">Additional features, such as analytics results will be combined with the LSTM results through a linear network, followed by classification output of categories that include exclusive tags “work”, “financial”, “social”, “shopping”, and one tag that can overlap with any of them, “important”. The network will look like the following:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 330.67px;">![](images/nntopology.png)</span>

<span class="c1"></span>

<span class="c1">While Google’s Word2Vec is the commonly referenced word embedding training method used presently, Stanford University has released a generally more accurate embedding model, called GloVe (global vectors) along with pre-trained embeddings for 400 thousand words, short phrases, and symbol combinations taken from a combination of Wikipedia and a large word database known as Gigawords. We will use Glove embeddings and test the number of dimensions required to get satisfactory results. Available pre-trained models for GloVe embeddings are 50, 100, 200, and 300 dimension embedding vectors. The tradeoff in vector size is generally accuracy of embedding, which translates in our case to accuracy of training, with training time, model size and compute time while running.</span>

# <span class="c2">Training Data</span>

<span class="c1">In order to provide an accurate classifier, we will need access to as much labeled training data as possible. We have discussed ways of getting data, which, for all categories besides “important”, can be acquired from objective sources in the market or from previously classified e-mails. Our determination of what is “important” is likely to require some assertions relative to the features we have defined, and we can then generate synthetic training data to train the concept of “important” into the neural network.</span>

<span class="c1"></span>

<span class="c1">It would also help if we could leverage all users and the current analytics plan by marketing the smart folders as “learning” folders, which really can learn in aggregate if people actively use them. Rather than collect PID from people who use the folders, we could relate some features we have defined above to the messages in each folder, especially to learn about how those features relate to what is “important” or possibly help exclude certain types of mail broadly.</span>

# <span class="c2">Implementation Notes</span>

<span class="c1">[miketout 9/19/2017]: I have implemented an initial neural network model, written a loader for the GloVe word vectors, a simple feature mapper, and an email parser/converter from email format to neural network input. As suggested in a meeting, I also integrated EFZP email parser for extracting body, signature, salutation, etc.</span>

<span class="c1"></span>

<span class="c1">I then ran some tests with a simple LSTM network, using the first 8 words of the subject and 22 words of the body to get a sense of the performance impact. Initially, classifying an email on an untrained network was taking a fully discounted time (including my own use of the machine, which was my notebook, so not super precise) of ~110 ms per email on a test of 1000 emails of the Enron dataset.</span>

<span class="c1">  
Since the EFZP parser was not strictly necessary, I wanted to determine how much overhead the neural networks were taking vs. the more standard parsing taking place in EFZP, so I tested without EFZP as well. I was surprised to learn that the time for each email dropped to 30ms, which seems pretty good for neural network analysis on a non-accelerated machine. If we replace the work done in separating salutation/signature, etc. by adding a few more words for neural network classification, I believe we would end up with better classification and performance as well. Based on this, I do not expect to use EFZP, though I may write some of my own parsing if it turns out to be helpful. These times were taken on a non-accelerated notebook with a 2.4Ghz Intel I7, quad core processor.</span>

<span class="c1"></span>

<span class="c1">I am currently looking for categorized email datasets that would be suitable to test the accuracy of the planned approach as well as the current neural networks architecture being used. While the current work quantifies the performance impact of the approach within an order of magnitude, we still don’t have data on the potential for accuracy.</span>

<span class="c1"></span>

<span class="c1">I have identified the following datasets for additional consideration. I have included one international email dataset as well, since there may be a way to identify languages using the current approach as well. I’ve also included a link to another email classification project using neural networks that also did more up front processing beforehand. I believe we are likely not to need that processing:</span>

<span class="c7">[http://bailando.sims.berkeley.edu/enron_email.html](https://www.google.com/url?q=http://bailando.sims.berkeley.edu/enron_email.html&sa=D&ust=1505953497843000&usg=AFQjCNHb6vpuEhwq5obDqhva-UiBIrGh8g)</span>

<span class="c7">[http://csmining.org/index.php/spam-email-datasets-.html](https://www.google.com/url?q=http://csmining.org/index.php/spam-email-datasets-.html&sa=D&ust=1505953497843000&usg=AFQjCNG-YKzHM-jloA2g3_LTS-8ImZrPxg)</span>

<span class="c7">[http://www.edrm.net/resources/data-sets/edrm-internationalization-data-set/](https://www.google.com/url?q=http://www.edrm.net/resources/data-sets/edrm-internationalization-data-set/&sa=D&ust=1505953497844000&usg=AFQjCNGTf9k8fyT9PWaO8cAWj0Lw5gHgEQ)</span>

<span class="c7">[http://www.andreykurenkov.com/writing/organizing-my-emails-with-a-neural-net/](https://www.google.com/url?q=http://www.andreykurenkov.com/writing/organizing-my-emails-with-a-neural-net/&sa=D&ust=1505953497844000&usg=AFQjCNE49A-ntJFQGUFGjpQFAfbI8p3sZg)</span>

<span class="c1"></span>

<span class="c1"></span>