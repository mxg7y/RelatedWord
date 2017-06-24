# coding: utf-8

import os, sys
import argparse
from gensim.models import word2vec
import MeCab

"""
    -w : 関連させたい単語
    -n : 関連させたくない単語
    -t : 表示する関連語の数
    -s : スコアも表示するかどうか
"""

parser = argparse.ArgumentParser(description="Display related words using word2vec")
parser.add_argument('-w', '--word', help='list of related words', action='store', nargs='*')
parser.add_argument('-n', '--nword', help='list of unrelated words', action='store', nargs='*')
parser.add_argument('-t', '--topn', help='num of displaying words', action='store', default=30)
parser.add_argument('-s', '--score', help='whether relation score is displayed', action='store_true')
args = parser.parse_args()

class Morph():
    def __init__(self):
        self.mcb = MeCab.Tagger('-Ochasen')
        self.stopwords = self.init_stopwords()

    def init_stopwords(self):
        stopwords = []
        with open('lib/stopwords.txt', 'r') as f:
            data = f.read()
            stopwords = [ word for word in data.split("\n") if word != '' ]
        return stopwords

    def filter_word_class(self, text, pass_class=[u"名詞", u"動詞"], shape='basic', stopword=True):
        node = self.mcb.parseToNode(text)
        filtered_words = []
        while node:
            word_class = node.feature.split(',')[0]
            basic_form = node.feature.split(',')[6]
            if word_class in pass_class and basic_form not in self.stopwords and basic_form != '*':
                if shape == 'basic':
                    filtered_words.append(basic_form)
                elif shape == 'surface':
                    filtered_words.append(node.surface)
            node = node.next
        return filtered_words

class Word2Vec():
    def __init__(self):
        self.w2v = word2vec.Word2Vec.load('model/word2vec.model')

    def related_word(self, pos_words, neg_words=[], top_n=30):
        pos_words = [ word for word in pos_words if self.in_vocabulary(word)]
        neg_words = [ word for word in neg_words if self.in_vocabulary(word)]
        if pos_words:
            return self.w2v.wv.most_similar(positive=pos_words, negative=neg_words, topn=top_n)
        else:
            return []

    def in_vocabulary(self, word):
        try:
            self.w2v.wv[word]
        except KeyError:
            print('Error: ' + word + 'はvocabularyに含まれていません')
            return None
        return word

if __name__=='__main__':

    '''
        Variables
    '''
    # standard input
    display_score = args.score
    display_num = int(args.topn)
    pos_words = args.word
    neg_words = args.nword

    # analyzer
    morph = Morph()
    w2v = Word2Vec()

    #
    if not pos_words:
        print('Error: 引数がありません')
        sys.exit()

    '''
        morphological analysis
    '''
    poslist = []
    neglist = []
    for word in pos_words:
        poslist.extend(morph.filter_word_class(word))
    if neg_words:
        for word in neg_words:
            neglist.extend(morph.filter_word_class(word))

    if not poslist:
        print('Error: 名詞か動詞を入力してください')
        sys.exit()

    '''
        similar word
    '''
    print('Positive words: ' + ' '.join(poslist))
    print('Negative words: ' + ' '.join(neglist))
    word_score = w2v.related_word(poslist, neg_words=neglist, top_n=display_num)
    if not word_score: sys.exit()
    if display_score:
        for w in word_score:
            print(w[0] + ': ' + str(w[1]))
    else:
        words = [ w[0] for w in word_score ]
        print(' '.join(words))
