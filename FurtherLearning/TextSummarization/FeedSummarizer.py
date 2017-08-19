import urllib2
from bs4 import BeautifulSoup
import re
import numpy as np
import networkx as nx
from gensim.summarization import summarize

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
sentence_tokenizer = PunktSentenceTokenizer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

from gensim.models import Word2Vec
model = Word2Vec.load("~/Desktop/ML_Summer2017/Kaggle_BagOfWords/300features_40minwords_10context")
index2word_set = set(model.wv.vocab)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english')
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

class Document (object) :

    def __init__ (self, title, text) :
        self.title = title
        self.text = text

    def preprocess_text (self) :
        wordlist = self.text.lower().split()
        wordlist = [w for w in wordlist if not w in stops]
        processed_text = " ".join(wordlist)

        processed_text = " ".join(processed_text.strip().split("\n"))
        sentences = sentence_tokenizer.tokenize(processed_text)
        sentences_original = sentence_tokenizer.tokenize(" ".join(self.text.strip().split("\n")))

        for i in xrange(0, len(sentences)) :
            sentences[i] = re.sub("[^a-zA-Z]", " ", sentences[i])

        return (sentences, sentences_original)

    def construct_SimilarityMatrix_WV (self, sentences) :
        sentence_matrix = []
        sentence_length = []

        num_features = 300

        for sentence in sentences :
            num_words = 0
            words = sentence.split()
            matrix = np.empty(shape=(len(words), 300), dtype=float)

            for word in words :
                if word in index2word_set :
                    matrix[num_words] = model[word]
                    num_words = num_words + 1

            extra_rows = len(words) - num_words

            for i in xrange(0, extra_rows) :
                matrix = np.delete(matrix, -1, 0)

            sentence_matrix.append(matrix)
            sentence_length.append(num_words)

        similarity_matrix_wv = np.identity(len(sentence_matrix))

        for i in xrange(0, len(sentence_matrix)) :
            for j in xrange(i+1, len(sentence_matrix)) :
                cum_sum = (np.dot(sentence_matrix[0], sentence_matrix[1].T)).sum(axis=0).sum()
                cum_sum = (2*cum_sum)/(sentence_length[i] + sentence_length[j])
                similarity_matrix_wv[i][j] = cum_sum
                similarity_matrix_wv[j][i] = cum_sum

        return similarity_matrix_wv

    def construct_SimilarityMatrix_TfIdf (self, sentences_original) :
        bow_matrix = vectorizer.fit_transform(sentences_original)
        transformed_matrix = tfidf_transformer.fit_transform(bow_matrix)
        similarity_matrix_ti = transformed_matrix * transformed_matrix.T

        return similarity_matrix_ti

    def summarize_myWay (self) :
        sentences, sentences_original = self.preprocess_text()
        similarity_matrix_wv = self.construct_SimilarityMatrix_WV(sentences)
        similarity_matrix_ti = self.construct_SimilarityMatrix_TfIdf(sentences_original)

        nx_graph_wv = nx.from_numpy_matrix(similarity_matrix_wv)
        scores_wv = nx.pagerank(nx_graph_wv)
        nx_graph_ti = nx.from_scipy_sparse_matrix(similarity_matrix_ti)
        scores_ti = nx.pagerank(nx_graph_ti)

        min_length = min(len(scores_ti), len(scores_wv))

        ranked = sorted((((scores_ti[i] + scores_wv[i])/2,i) for i in xrange(0, min_length)), reverse=True)

        summary_length = max(len(sentences)/10, 10)
        ordered_summary_indices = sorted(ranked[i][1] for i in xrange(0, summary_length))
        summary = []

        print self.title
        for i in ordered_summary_indices :
            print sentences_original[i]

        return

    def summarize_usingPackage (self) :
        sentences_original = sentence_tokenizer.tokenize(" ".join(self.text.strip().split("\n")))
        text_reconstructed = " ".join(sentences_original)
        package_summary = summarize(text_reconstructed)
        print self.title
        print package_summary

        return


def get_text_only (article_url) :
    page = urllib2.urlopen(article_url).read().decode('utf8')
    soup = BeautifulSoup(page)
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return soup.title.text, text

def summarizeFeed (feed_url, num_articles, package_version=False) :
    feed_xml  = urllib2.urlopen(feed_url).read()
    feed = BeautifulSoup(feed_xml.decode('utf8'))
    to_summarize = map(lambda p: p.text, feed.find_all('guid'))

    print ("\n")
    print "Summary of your feed .... "
    print ("\n")

    for article_url in to_summarize[:num_articles] :
        article_title, article_text = get_text_only(article_url)
        article_doc = Document(article_title, article_text)

        if package_version :
            article_doc.summarize_usingPackage()
        else :
            article_doc.summarize_myWay()
        print '-----------------------------------------------------------------'

    return
