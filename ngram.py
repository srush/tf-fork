#!/usr/bin/env python

import sys
logs = sys.stderr

import srilm

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_string("lm", None, "SRILM language model file")
flags.DEFINE_integer("order", 3, "language model order")

class Ngram(object):
    '''a class of variables and methods (served as globals for other modules).'''

    @staticmethod
    def cmdline_ngram():
        if FLAGS.lm is None:
            return None
#             print >> logs, "Error: must specify an LM file --lm" + str(FLAGS)
#             sys.exit(1)            
        return Ngram(order=FLAGS.order, lmfilename=FLAGS.lm)
    
    def __init__(self, order, lmfilename):
        self.order = order
        self.vocab = srilm.Vocab(True)
        self.ngram = srilm.Ngram(self.vocab, order)

        import time
        t0 = time.clock()
        print >> logs, "reading lm from", lmfilename
        print >> logs, "lm order is %d"  % order
        self.ngram.read(lmfilename, limit_vocab=False)
        print >> logs, "lm loaded in %.2lf seconds" % (time.clock() - t0)
        self.pqcache = {}

        self._stopsyms = [self.vocab.index("</s>")] * (self.order - 1)
        self._startsyms = [self.vocab.index("<s>")] * (self.order - 1)
        
    def word_prob(self, s):
        ## traditional interface (for debugging), although i can use pq()

        news = "<s> " * (self.order - 1) + s + " </s>"
        t = self.words2indices(news)
        score = 0
        for i in range(self.order - 1, len(t)):
            score += self.ngram.wordprob(t[i], t[i - self.order + 1: i])
        return -score   # negative logprob

    def word_prob_bystr(self, s, his):
        ## traditional interface (for debugging), although i can use pq()
        ns = self.word2index(s)
        nhis = self.words2indices(his)
        score = self.ngram.wordprob(ns, nhis)
        return score   # negative logprob

        
    def clear(self):
        self.pqcache = {}
        
    def _pq(self, qs):
        '''real work for pq'''
        l = len(qs)

        if l >= self.order:
            q = qs[:self.order - 1] + (-1,) + qs[l - self.order + 1:]
        else:
            q = qs[:] ### caution!!! slicing to copy!

#       if settings.debugLM:
#           print >> logs, "evaluating %s" % self.ppqstr(qs)
        p = 0
        i = self.order - 1
        while i < l:
            if qs[i] == -1:  # meet a *, skip m grams
                i += self.order 
            else:
                lm_prob = -self.ngram.wordprob(qs[i], qs[i - self.order + 1: i])  # make it positive
                p += lm_prob
                i += 1
                
        if settings.debugLM:
            print >> logs, "p = %.3lf, q = %s" % (p, self.ppqstr(q))

        return p, qstr.QStr(q)

    def pq(self, qs):
        '''s can be any valid *-string
        return (p, q)
        '''

        res = self.pqcache.get(qs, None)
        if res is not None:
            return res

        p, q = self._pq(qs)
        self.pqcache[qs] = (p, q)
        return p, q

    def rawpq(self, ws):
        return self.pq(self.words2indices(ws))
    
    def word2index(self, w):
        return self.vocab.index(w)

    def words2indices(self, ws):
        '''mapping a list of words (strings) to lm indices
        return a tuple (instead of a list)
        '''
        if type(ws) == str:
            ws = ws.split()
        return tuple([self.word2index(w) for w in ws])

    def index2word(self, qitem):
        if qitem == -1:
            return "*"
        return self.vocab.word(qitem)

    def ppqstr(self, qs):
        return " ".join([self.index2word(qitem) for qitem in qs])

    def pre(self, qs):
        p = 0
        for i in range(min(len(qs), self.order - 1)):
            p += -self.ngram.wordprob(qs[i], qs[:i])  # make it positive
        return p

    def startsyms(self):
        ''' <s>^{order - 1} '''
        return self._startsyms

    def stopsyms(self):
        ''' </s> '''
        return self._stopsyms

    def raw_startsyms(self):
        ''' <s>^{order - 1} '''
        return ["<s>"] * (self.order - 1)

    def raw_stopsyms(self):
        ''' </s> '''
        return ["</s>"] * (self.order - 1) # not single <s>, so that DP can stop at unique state
