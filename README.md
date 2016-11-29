# The Machine Learning NLP Bible

# Major Tasks in NLP
The following is a list of some of the most commonly researched tasks in NLP. They are split up into two categories: **tasks** that have direct real-world applications and **sub-taskes** that are used to aid in solving larger tasks.

**Real Life Tasks:**
* Automatic Summarization
*

**Sub-tasks:**
* Named entity recognition
* Part of speech tagging
*
* Discourse analysis
* Co-reference analysis
* Morphological Analysis
* Relationship extraction
* Sentence breaking


## Tasks

## Subtasks



# Word Vectors


Vector space models (VSMs) represent (embed) words in a continuous vector space where semantically similar words are mapped to nearby points ('are embedded nearby each other'). VSMs have a long, rich history in NLP, but all methods depend in some way or another on the Distributional Hypothesis, which states that words that appear in the same contexts share semantic meaning. The different approaches that leverage this principle can be divided into two categories: count-based methods (e.g. Latent Semantic Analysis), and predictive methods (e.g. neural probabilistic language models).Printy. online printing

<center>![Alt text](/Users/robsalz/.atom/evnd/tmp/clipboard_20160629_144303.png "Optional title")
## Word vector models
[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013.
Generate word and phrase vectors.  Performs well on word similarity and analogy task and includes [Word2Vec source code](https://code.google.com/p/word2vec/)  Subsamples frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)

[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
Chris Olah (2014)  Blog post explaining word2vec.

[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/)

# Text Classification
The process of converting a sequence of of natural language tex to a discrete class. The text classification task can be broken up into three categories, with increasing degrees of difficulty.

1. Binary: Classify into one of two classes
2. Multi-class: Classify multiple into multiple distinct classes
3. Multi-label: Classify one document into multiple, overlapping classes

Applications include:
* Sentiment (binary: _positive | negative_; fine grained: _very neg | neg | neutral | pos | very pos_ )
* Classifying emails (spam|not spam), (type of email);
* Classifying types of questions
* Document topic classification (News articles, Wikipedia pages, )


## Text classification - Models:
* [A Convolutional Neural Network for Modelling Sentences](http://arxiv.org/pdf/1404.2188v1.pdf)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)
* [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ):
* [Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)
* [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)
* [Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)
* [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
* [A C-LSTM Neural Network for Text Classification](http://arxiv.org/pdf/1511.08630v2.pdf)
* [Deep LSTM based Feature Mapping for Query Classification](http://m-mitchell.com/NAACL-2016/NAACL-HLT2016/pdf/N16-1176.pdf)

## Text classification - Datasets:
| <sub>Name                                                                                                                                                                   | <sub>Type                                                      | <sub>#Classes | <sub>#Documents | <sub>Format     | <sub>Description                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|---------------|-----------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <sub>Stanford Sentiment Treebank*: [Paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf); [Download](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) | <sub>**Sentiment**: Movie reviews                                  | <sub>5        | <sub>11,855     | <sub>.txt       | <sub>The movie reviews with five classes (negative, somewhat negative, neutral, somewhat positive, positive)                                                  |
| <sub>IMDB: [Paper](http://ai.stanford.edu/~amaas/data/sentiment/); [Download](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)                               | <sub>**Sentiment**: Movie reviews                                  | <sub>2        | <sub>100,000    |                 | <sub>The IMDB dataset2 consists of 100,000 movie reviews with binary classes. One key aspect of this dataset is that each movie review has several sentences. |
| <sub>SUBJ: [Paper](http://www.cs.cornell.edu/home/llee/papers/cutsent.pdf); [Download](http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz)          | <sub> **Type**: Objective/Subjective Movie Description | <sub>2        | <sub> 10,000    | <sub>.txt       | <sub>Subjectivity data set where the goal is to classify each instance (snippet) as being subjective or objective.                                            |
| <sub>TREC: [Page](http://cogcomp.cs.illinois.edu/Data/QA/QC/); [Download](http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label)                                       | <sub> **Categories**: Question type                                | <sub>6        | <sub>5,452      | <sub>.txt       | <sub>List of short questions, divided into 6 categories, including location, human, entity, abbreviation, description and numeric                             |
| <sub>20 News Groups: [Page](http://qwone.com/~jason/20Newsgroups/); [Download](http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz)                                    | <sub>**Categories**: News                                          | <sub>20       | <sub>18,828     | <sub>.txt files | <sub>The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics                                                                           |
| <sub>Yelp dataset challenge*:  [Page](https://www.yelp.com/dataset_challenge)                                                                                                    | <sub>**Sentiment**: Restaurant  reviews                            | <sub>5        | <sub>280,000    | <sub>.json       | <sub>The Yelp Dataset Challenge data is in rich json format with business info, checkins, images and reviews.                                                 |
## Text classification - Standardizing dataset formats
There are a lot of different corpora for different text classification tasks. We propose, and a lot of time is spent handling the data to fit your model, or restructuring your model to fit the data. We have decided to generalize and then standardize the process.

Lets deconstruct the format a document dataset can be in. The dataset should be iterable over documents and targets. The documents should be iterable over characters or words. If its words, you need to define a tokenizing method, common to use spaces to separate out words.

To represent a document, we need a sequence of vectors. The vector can be of the form:
1. Hot encoded characters
2. Hot encoded words _(very sparse)_
3. Word vectors

Not efficient to store full sequence of vectors, so we store lookup dictionaries, and a sequence of indexes instead:

Database for text classification tasks and language models
```json
collection_dataset:{
  "count_dict":{"UNK":2827,"and":12312,"the":28381},
  "word_dictionary": {"1":"aa","2":"acorn","...":"...","50000":"zebra"},
  "word_dictionary_reversed": {"aa":1},

  "char_dictionary":{"1":"a","2":"b"},
  "char_dictionary_reversed": {"a":1,"b":2},

  "document":{"_id" : "123124",
              "number": 1,
              "raw" : "aa and in went to the shop to find green eggs",
              "word_idx_document": "1 32 41 2421",
              "char_idx_document": "1 21 23 12 0 12 3 18 0",
              "word_length": 11,
              "char_length": 35,
              "class":"c12",
              "bucket":"1"
  },
  "document":{"_id" : "1d9s124",
              "number": 2,
              "raw" : "aa and in went to the shop to find green eggs",
              "word_idx_document": "1 32 41 2421",
              "char_idx_document": "1 21 23 12 0 12 3 18 0",
              "word_length": 11,
              "char_length": 35,
              "class":"c12"
  }
}

```
Note, when training a neural network you are going to want to create batches of inputs and targets. This is general for any task. To speed up the process, you want texts of similar length to be placed into a bucket. The bucket will then be sampled randomly for a batch.

Question is do you want to query the database once, or multiple time. What do you want to store in memory...



# Named entity recognition

# Deep Learning for NLP resources

Introductory and state of the art resources for NLP sequence modeling tasks like dialog.
## Machine Learning: Neural Networks, RNN, LSTM
[Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome?module=tN10A)
Andrew Ng
Introductory course for linear regression, logistic regression, and neural networks.
Also covers support vector machines, k-means, etc.

[Cousera: Neural Networks](https://class.coursera.org/neuralnets-2012-001/lecture)
[Geoffrey Hinton](https://scholar.google.com/citations?user=JicYPdAAAAAJ)
Covers a variety of topics: Neural nets, RNNs, LSTMs.

[Machine Learning for Developers](http://xyclade.github.io/MachineLearning/)
Intro to basic ML concepts for developers.

[Deep Learning (Book)](http://goodfeli.github.io/dlbook/)
[Yoshua Bengio](https://scholar.google.com/citations?user=kukA0LcAAAAJ&hl=en)
Advanced book about deep learning.

[A few useful things to know about machine learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
Pedro Domingos

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
Blog post by Chris Olah.

## Deep Learning for NLP
[Stanford Natural Language Processing](https://class.coursera.org/nlp/lecture/preview)
Intro NLP course with videos. This has no deep learning. But it is a good primer for traditional nlp.

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)
[Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). (2016)  Class with syllabus, and slides.
Videos: [2015 lectures] (https://www.youtube.com/channel/UCsGC3XXF1ThHwtDo18d7WVw/videos) 2016: lecture [1](https://www.youtube.com/watch?v=kZteabVD8sU&index=1&list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi) [2](https://www.youtube.com/watch?v=xhHOL3TNyJs&feature=youtu.be) [3](https://www.youtube.com/watch?v=UOGMsFw9V_w&feature=youtu.be) [4](https://www.youtube.com/watch?v=bjDbNbSbwY4&feature=youtu.be) [5](https://www.youtube.com/watch?v=k50GPWfjG7I&feature=youtu.be) [6](https://www.youtube.com/watch?v=l0k-30FNua8&feature=youtu.be) [7] (https://www.youtube.com/watch?v=L8Y2_Cq2X5s&list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi&index=8) [8](https://www.youtube.com/watch?v=nwcJuGuG-0s&index=7&list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi)

[A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)
Yoav Goldberg. October 2015. No new info, 75 page summary of state of the art.



# Machine Translation
[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf)
Bahdanau, Cho 2014.  "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  Implements attention mechanism.
[English to French Demo](http://104.131.78.120/)

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses LSTM RNNs to generate translations. " Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8"
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in [TensorFlow](http://tensorflow.org/).


# Written dialog systems
[A Neural Network Approach toContext-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)
Sordoni 2015.  Generates responses to tweets.
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.

[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine transation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)

### Written dialog datasets
| <sub>Name                                                                                                                                                                                                                                                             | <sub> Type        | <sub>Topics        | <sub>Avg turns | <sub> Dialogs | <sub> Size         | <sub>Description                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|--------------------|----------------|---------------|--------------------|-----------------------------------------------------------------------------------------------------------|
| <sub>NPS Chat Corpus [Paper](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4338328&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4338328); [Download](https://catalog.ldc.upenn.edu/LDC2010T05)                                            | <sub>Chat         | <sub> Unrestricted | -              | -             | <sub>100M words    | <sub>Posts from age-specific online chat rooms                                                            |
| <sub>Twitter triple corpus [Paper](http://research.microsoft.com/en-us/downloads/6096d3da-0c3b-42fa-a480-646929aa06f1/); [API-calls](http://ftp.research.microsoft.com/downloads/6096d3da-0c3b-42fa-a480-646929aa06f1/MSRSocialMediaConversationCorpus.zip) |  <sub> Microblog  | <sub>Unrestricted  | <sub>3         | <sub>4,232         | -                  | <sub>A collection of Tweet Ids representing threestep conversational snippets extracted from Twitter logs |
| <sub>UseNet Corpus [Paper](); [Download]()                                                                                                                                                                                                                            | <sub>Microblog    | <sub>Unrestricted  | -              | -             | <sub>7B words      | <sub>UseNet forum postings                                                                                |
| <sub>NUS SMS Corpus [Paper](http://wing.comp.nus.edu.sg:8080/SMSCorpus/pubs.jsp); [Download](http://wing.comp.nus.edu.sg:8080/SMSCorpus/)                                                                                                                                  | <sub>SMS messages | <sub>Unrestricted  | -              | -             | <sub>580,668 words | <sub>SMS messages collected between two users, with timing analysis.                                      |
| <sub>Reddit - Change my view [Paper](https://chenhaot.com/pages/changemyview.html); [Download](https://chenhaot.com/data/cmv/cmv.tar.bz2)                                                                                                                                  | <sub>Forum        | <sub>Unrestricted  | -              | -             | <sub>321MB         | <sub>Metadata, and related data from the subreddit ChangeMyView                                           |
| <sub>Settlers of Catan [Paper](https://hal.inria.fr/hal-00750618/document);                                                                                                                                                                                                | <sub>Chat         | <sub>Game terms    | <sub>95             | <sub>21            | -                  | <sub>Conversations between players in the game ‘Settlers of Catan’                                             |
| <sub> Ubuntu dialog corpus [Paper](http://nldslab.soe.ucsc.edu/iac/iac_lrec.pdf); [Request download](https://nlds.soe.ucsc.edu/iac)                                                                                                                                                                | <sub> Forum       | <sub> Politics     | <sub>34.5           | <sub>11K           | <sub>73M                | <sub>The Internet Argument Corpus (IAC) is a corpus for research in political debate on internet forums        |

## Single exchange Dialog


## Factoid question answering

## Memory and attention models
Attention mechanisms allows the network to refer back to the input sequence, instead of forcing it to encode all information into one fixed-length vector.
* [Attention and Memory in Deep Learning and NLP](http://www.opendatascience.com/blog/attention-and-memory-in-deep-learning-and-nlp/)
* [Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf)
* [End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf)
* [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)
* [Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)
* [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)
* [Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)
* [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)
* [Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)
* Dynamic Memory Network ([Paper 1](), [Paper 2]()):
![Alt text](/Users/robsalz/.atom/evnd/tmp/clipboard_20160810_130324.png "Optional title")

## Task orientated dialog
* [End-to-end LSTM-based dialog control optimized with
supervised and reinforcement learning](https://arxiv.org/pdf/1606.01269v1.pdf)
![Alt text](/Users/robsalz/.atom/evnd/tmp/clipboard_20160628_225900.png "Optional title")


## Courses
## Companies
* [Digital genius](https://digitalgenius.com/): Automate answers to customer service questions
* [Sentisis](http://sentisis.com/en/): Create business solutions based on semantic analysis of Spanish conversations
* [Netbase](http://www.netbase.com/): Social Media Analysis
* [Swiftkey](https://swiftkey.com/en): Text Prediction
* [Fiscalnote](https://www.fiscalnote.com/): Predicting Government Legislation
* [Mindmeld](https://mindmeld.com/):  Build tools that enable companies to create intelligent voice-driven interfaces for any app or device
* [Vurb](): Mobile Search, Messaging, and Contextual Layer for Apps
* [ReTargeter](): Making targeted display advertising easy and effective
* [Klevu](http://www.klevu.com/features.html): Smart search for shopping
* [x.Ai](): Aakes an artificial intelligence personal assistant who schedules meetings for you
